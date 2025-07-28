import javalang
from javalang.parser import JavaSyntaxError

# 找结束行
def find_code(start_line,code):
    code_lines = code.splitlines()
    relevant_code = code_lines[start_line:]
    end_line = start_line-1
    bracket_left = 0
    bracket_right = 0
    for line in relevant_code:
        end_line += 1
        stripped_line = line.strip()
        for char in stripped_line:
            if char == '{':
                bracket_left += 1
            elif char == '}':
                bracket_right += 1
        if 0 < bracket_left <= bracket_right:
            break
    return end_line

# 找函数参数
def find_parameters(start_line,code):
    code_lines = code.splitlines()
    bracket_count = 0
    full_signature = []
    current_line = start_line
    while current_line<len(code_lines):
        line = code_lines[current_line]
        relevant_part = line.strip()
        for char in relevant_part:
            if bracket_count == 0 and ' {' in relevant_part and current_line !=start_line:
                break

            if char == '(':
                bracket_count += 1
            elif char == ')':
                bracket_count -= 1
            if bracket_count>0:
                full_signature.append(char)

            # 如果括号匹配完成并且遇到了 '{'，则停止读取
            if bracket_count == 0 and char == ')':
                break
        # 如果括号匹配完成并且遇到了 '{'，则停止读取
        if bracket_count == 0 and ' {' in relevant_part:
            break

        # 更新下一行
        current_line += 1
    full_signature_str = ''.join(full_signature[1:])
    # print(full_signature_str)
    return full_signature_str

# 找函数返回值
def find_return_type(start_line,code,modifiers):
    code_lines = code.splitlines()
    line = code_lines[start_line]
    relevant_part = line.strip()
    type_start = False
    return_type = []
    find_modifiers = False
    if modifiers == '':
        find_modifiers = True
    ##如果识别到空格，且已经在前面找到函数修饰符，并且已经明确第二个非空字符已经出现即返回类型已经全部提取
    for char in relevant_part:
        if char == " ":
            if type_start:
                break
            else:
                continue
        if find_modifiers:
            return_type.append(char)
            type_start =True
            continue
        if char == modifiers:
            find_modifiers = True
    return ''.join(return_type)

# 将代码分割成函数
def extract_methods_from_java_code(java_code_str):
    # 解析Java源代码为抽象语法树
    try:
        # 尝试解析Java代码
        tree = javalang.parse.parse(java_code_str)
        # print("解析成功")
    except JavaSyntaxError as e:
        # 捕获Java语法错误
        print(f"Java语法错误: {e.description}")
        print(f"错误发生在:  {e.at}")

    methods = []
    # 遍历抽象语法树中的所有节点
    for path, node in tree:
        # 检查节点是否是方法声明或构造函数
        if isinstance(node, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
            start_line = node.position.line
            parameters = find_parameters(start_line-1,java_code_str)
            modifiers = ' '.join(node.modifiers)


            if isinstance(node, javalang.tree.ConstructorDeclaration):
                return_type = None
            else:
                if node.return_type:
                    # return_type = node.return_type.name
                    return_type = find_return_type(start_line,java_code_str,modifiers)
                else:
                    return_type = 'void'

            signature = f"{node.name}({parameters})"
            end_line = find_code(start_line-1,java_code_str)

            # 提取方法的代码块
            method_code = '\n'.join(line for line in java_code_str.splitlines()[start_line - 1:end_line+1])
            # 添加到方法列表中
            methods.append({
                'name':node.name,
                'parameters':parameters,
                'modifiers':modifiers,
                'signature':signature,
                'code': method_code,
                'start_line': start_line,
                'end_line': end_line+1
            })
    return methods

def printMethod(methods):
    for method in methods:
        print(f"Method: {method['name']}({method['parameters']})")
        print("Code:")
        print(method['code'])
        print('---')
