import re

def strip_leading_whitespace(line):
    # 使用正则表达式去除行首的空白字符
    return re.sub(r'^\s+', '', line)

def process_diff(diff_lines):
    # 初始化三个列表来存储不同版本的代码
    removed_symbols = []
    before_change = []
    after_change = []

    # 初始化两个字典来记录增加和删除的行信息
    added_lines = {}
    deleted_lines = {}

    # 记录当前行号
    original_line_number = 0
    add_line_number = 0
    delete_line_number = 0

    for line in diff_lines:
        # original_line_number += 1
        add_line_number += 1
        delete_line_number += 1
        if line.startswith('+') and not line.startswith('+++'):
            # 如果是增加的行，添加到变更后的版本和去除标识的版本中，但不包括 '+' 符号
            delete_line_number -= 1
            clean_line = ' '+line[1:]
            stripped_line = strip_leading_whitespace(clean_line)
            after_change.append(clean_line)
            removed_symbols.append(clean_line)
            added_lines[stripped_line] = add_line_number
            deleted_lines[stripped_line] = delete_line_number
        elif line.startswith('-') and not line.startswith('---'):
            # 如果是删除的行，添加到变更前的版本和去除标识的版本中，但不包括 '-' 符号
            add_line_number -= 1
            clean_line =' '+ line[1:]
            before_change.append(clean_line)
            removed_symbols.append(clean_line)
            # deleted_lines[stripped_line] = original_line_number
        else:
            # 对于其他行（例如上下文），添加到所有三个版本中
            removed_symbols.append(line)
            before_change.append(line)
            after_change.append(line)

    return (
        '\n'.join(removed_symbols),
        '\n'.join(before_change),
        '\n'.join(after_change),
        added_lines,
        deleted_lines
    )

