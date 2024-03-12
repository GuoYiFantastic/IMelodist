import re
def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配中文字符的正则表达式范围
    return bool(pattern.search(text))
def contains_comment(text):
    pattern = re.compile(r'\[\d+\]')
    return bool(pattern.search(text))

with open('./data/beethoven_no_space_8.txt','r', encoding='utf-8') as fr:
    lines = fr.readlines();
    with open('./data/beethoven_no_space_9.txt', 'w', encoding='utf-8') as fw:
        for line in lines:
            if not line.startswith('('):
                fw.write(f'{line.replace("（","《").replace("）","》")}')
