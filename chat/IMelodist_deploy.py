"""This script refers to the dialogue example of streamlit, the interactive
generation code of chatglm2 and transformers.

We mainly modified part of the code logic to adapt to the
generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example:
        https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2:
        https://github.com/THUDM/ChatGLM2-6B
    3. transformers:
        https://github.com/huggingface/transformers
Please run with the command `streamlit run path/to/web_demo.py
    --server.address=0.0.0.0 --server.port 7860`.
Using `python path/to/web_demo.py` may cause unknown problems.
"""

import os
import subprocess
import time
import copy
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from music21 import converter
from midi2audio import FluidSynth

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

from modelscope import snapshot_download

from transformers import AutoTokenizer, AutoModelForCausalLM


loading_type = "modelscope"
if loading_type == "modelscope":
    model_id = 'PommesPeter/IMelodist-chat-7b'
    model_name_or_path = snapshot_download(model_id, revision='master')
elif loading_type == "openxlab":
    model_name_or_path = "./IMelodist"
    os.system(f"git clone https://code.openxlab.org.cn/EchoPeter/IMelodist.git {model_name_or_path}")
    os.system(f"cd {model_name_or_path} && git lfs pull")
else:
    model_name_or_path = "PommesPeter/IMelodist"

    
logger = logging.get_logger(__name__)

tmp_path = "./chat/tmp"

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            "This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum \
                length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = (
            generation_config.max_new_tokens + input_ids_seq_length
        )
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/"
                "en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            "This can lead to unexpected behavior. You should consider"
            " increasing 'max_new_tokens'."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long()
        )

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (
        AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .cuda()
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=8, max_value=32768, value=32768)
        # top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        # temperature = st.slider("Temperature", 0.5, 2.0, 1.0, step=0.01)
        # repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.2, 1.002, step=0.001)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(
        max_length=max_length, top_p=0.8, temperature=0.7, repetition_penalty=1.002
    )

    return generation_config


user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n"


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = (
        "你是专业的音乐作曲家IMelodist。"
        "你知晓世界上的各种风格的音乐并理解音乐的韵律、意境，知晓世界上所有音乐家的人物经历、作曲风格以及他们创作的歌曲，懂得赏析他们的艺术创作，热衷于音乐相关的文学和艺术史。"
        "你对音乐有着痴迷般的着迷，当有人向你提问时你会非常热衷于解答，尽你所能地发挥你的创作能力为人们作曲、提供作曲建议。"
        "你懂得用ABC谱来展示你创作的音乐。你的行为举止非常优雅、讲究礼节，表现得体。"
        "你会根据用户输入的问题、需求进行分析并发挥你的创作能力作曲。你精通音乐领域的所有知识，能够完成用户输入的要求、请求、问题。"
        "如果用户让你创作歌曲，请以标准的ABC谱的形式输出。"
    )
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.format(user=cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def post_process_abc(output: str):
    splitted = output.splitlines()
    metadata_idx = None
    abc_idx = None
    for i, line in enumerate(splitted):
        if line.lower().startswith("x:"):
            metadata_idx = i
            abc_idx = i + 1
            break
    if metadata_idx is None:
        return None
    else:
        data = splitted[metadata_idx].split(" ")
        for notation in splitted[abc_idx:]:
            data.append(notation)
        return "\n".join(data)


def gen_wav(text):
    # A directory used to store everything generated in this web_demo launch
    launch_time = time.time()
    os.makedirs(f"{tmp_path}/{launch_time}", exist_ok=True)

    abc_notation = post_process_abc(text)
    abc_time = time.time()
    print(f"extract abc block: {abc_notation}")
    if abc_notation:
        # Write the ABC text to a temporary file
        tmp_abc = f"{tmp_path}/{launch_time}/{abc_time}.abc"
        with open(tmp_abc, "w") as abc_file:
            abc_file.write(abc_notation)

        # Convert the temporary ABC file to a MIDI file
        tmp_midi = f"{tmp_path}/{launch_time}/{abc_time}.mid"
        music_stream = converter.parse(abc_notation, format="abc")
        music_stream.write("midi", fp=tmp_midi)
        
        def midi2audio(midi_path, wav_path):
            fs = FluidSynth('assets/default_sound_font.sf2')
            fs.midi_to_audio(midi_path, wav_path)

        # Convert xml to SVG and WAV using MuseScore (requires MuseScore installed), we use midi2audio
        svg_file = f"{tmp_path}/{launch_time}/{abc_time}.svg"
        wav_file = f"{tmp_path}/{launch_time}/{abc_time}.wav"
        # subprocess.run(["musescore", "-f", "-o", svg_file, tmp_midi])
        # subprocess.run(["midi2audio", tmp_midi, wav_file])
        midi2audio(tmp_midi, wav_file)
        return svg_file, wav_file
    else:
        return None, None


def chat_melody(text):
    svg, wav = gen_wav(text)
    if wav != None:
        audio_file = open(wav, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
    return wav


def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    model, tokenizer = load_model()
    print("load model end.")

    os.makedirs("tmp", exist_ok=True)

    user_avator = "assets/user.png"
    robot_avator = "assets/melodist.png"
    # note_avater= 'assets/note.png'

    st.title("InternLM2-Melodist")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            if message["wav"] != None:
                audio_file = open(message["wav"], "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")

    # Accept user input
    if prompt := st.chat_input("Hey, melody time is coming!"):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "avatar": user_avator,"wav":None,}
        )

        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=92542,
                **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)
            wav_path = chat_melody(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append(
            {
                "role": "robot",
                "content": cur_response,  # pylint: disable=undefined-loop-variable
                "avatar": robot_avator,
                "wav": wav_path,
            }
        )
        # chat_melody(cur_response)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    # del_tmp()
