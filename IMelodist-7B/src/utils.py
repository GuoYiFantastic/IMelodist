

# test = """当然，我会尽我所能为您创作一首简短而优美的乐曲。以下是基于A大调的简单钢琴曲，它充满了活力与轻快的节奏，适合作为背景音乐或个人练习使用。希望您会喜欢它：
# X:163928 L:1/8 M:4/4 K
# |: E2 | A3 B cBcA | BAAB cBAG | F2 AF d2 cd | edcB AGEG | A3 B cBcA | BAAB cBAG | f2 ed gdcB | A2 AA A2 :: a2 | agfg efed | egfa gfef | defg afge | dcde fgaf | agfe dfed | egfa gfeg | fafd gdcB | A2 AA A2 :|
# """
def post_process(output: str) -> str:
    splitted = output.splitlines()
    formatted_abc_notation = splitted[1].split(' ')
    formatted_abc_notation.append(splitted[2])
    return '\n'.join(formatted_abc_notation)
    
# abc = post_process(test)
# print(abc)
