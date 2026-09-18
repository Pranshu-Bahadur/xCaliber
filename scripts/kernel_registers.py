#!/usr/bin/env python3

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path


SCALAR_SLOTS = {
    "bool": 1, "char": 1, "signed char": 1, "unsigned char": 1,
    "short": 1, "unsigned short": 1, "int": 1, "unsigned": 1,
    "unsigned int": 1, "long": 2, "unsigned long": 2,
    "long long": 2, "unsigned long long": 2, "float": 1, "double": 2,
    "int8_t": 1, "uint8_t": 1, "int16_t": 1, "uint16_t": 1,
    "int32_t": 1, "uint32_t": 1, "int64_t": 2, "uint64_t": 2,
    "size_t": 2, "__half": 1, "half": 1, "__nv_bfloat16": 1,
    "nv_bfloat16": 1, "dim3": 3,
}

VECTOR_BYTES = {
    "char": 1, "uchar": 1, "short": 2, "ushort": 2, "half": 2,
    "bfloat16": 2, "int": 4, "uint": 4, "float": 4,
    "longlong": 8, "ulonglong": 8, "double": 8,
}


@dataclass
class Variable:
    line: int
    type: str
    name: str
    shape: str
    slots: int | None
    note: str = ""


@dataclass
class Kernel:
    name: str
    body: str
    body_offset: int


def mask_noncode(source: str) -> str:
    out = list(source)
    i, state = 0, "code"
    while i < len(source):
        pair, char = source[i:i + 2], source[i]
        if state == "code":
            if pair in {"//", "/*"}:
                out[i] = out[i + 1] = " "
                i += 2
                state = "line" if pair == "//" else "block"
                continue
            digit_separator = (
                char == "'" and i > 0 and i + 1 < len(source)
                and source[i - 1].isalnum() and source[i + 1].isalnum()
            )
            if char in {'"', "'"} and not digit_separator:
                out[i] = " "
                state = "string" if char == '"' else "char"
        elif state == "line":
            if char == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block":
            if pair == "*/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
                continue
            if char != "\n":
                out[i] = " "
        else:
            out[i] = " "
            if char == "\\" and i + 1 < len(source):
                if source[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        i += 1
    return "".join(out)


def find_kernels(source: str) -> list[Kernel]:
    code = mask_noncode(source)
    kernels = []
    for marker in re.finditer(r"\b__global__\b", code):
        brace = code.find("{", marker.end())
        if brace < 0:
            continue
        calls = list(re.finditer(r"([A-Za-z_]\w*)\s*\(", code[marker.start():brace]))
        if not calls:
            continue
        depth, end = 1, brace + 1
        while end < len(code) and depth:
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        if depth == 0:
            kernels.append(
                Kernel(calls[-1].group(1), code[brace + 1:end - 1], brace + 1)
            )
    return kernels


def eval_integer(expression: str, constants: dict[str, int]) -> int | None:
    expression = expression.replace("'", "")
    expression = re.sub(r"(?<=\d)[uUlL]+\b", "", expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    operators = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a // b,
        ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b,
        ast.BitOr: lambda a, b: a | b, ast.BitAnd: lambda a, b: a & b,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError

    try:
        return int(visit(tree))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def type_slots(type_name: str, pointer: bool) -> int | None:
    if pointer:
        return 2
    type_name = " ".join(type_name.split())
    if type_name in SCALAR_SLOTS:
        return SCALAR_SLOTS[type_name]
    vector = re.fullmatch(
        r"(char|uchar|short|ushort|half|bfloat16|int|uint|float|longlong|ulonglong|double)([1-4])",
        type_name,
    )
    if vector:
        return math.ceil(VECTOR_BYTES[vector.group(1)] * int(vector.group(2)) / 4)
    return None


def split_declarators(text: str) -> list[str]:
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return parts + [text[start:]]


def declarations(kernel: Kernel, source: str) -> list[Variable]:
    constants = {}
    for name, value in re.findall(r"^\s*#define\s+(\w+)\s+([^\n]+)", source, re.MULTILINE):
        number = eval_integer(value, constants)
        if number is not None:
            constants[name] = number

    pattern = re.compile(
        r"(?:^|[;{}(])\s*"
        r"(?P<qual>(?:(?:const|constexpr|volatile|register|static|__restrict__|__shared__|extern)\s+)*)"
        r"(?!asm\b|if\b|for\b|while\b|return\b)"
        r"(?P<type>(?:unsigned|signed)\s+(?:char|short|int|long(?:\s+long)?)|"
        r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*(?:\s*<[^;=]+?>)?)\s+"
        r"(?P<rest>[^;]+)",
        re.DOTALL,
    )
    variables = []
    for match in pattern.finditer(kernel.body):
        qualifiers = set(match.group("qual").split())
        type_name = " ".join(match.group("type").split())
        if type_name in {"asm", "if", "for", "while", "return"}:
            continue

        for item in split_declarators(match.group("rest")):
            parsed = re.match(
                r"\s*(?P<prefix>[*&\s]*)(?P<name>[A-Za-z_]\w*)\s*"
                r"(?P<shape>(?:\[[^\]]*\]\s*)*)",
                item,
            )
            if not parsed:
                continue
            name = parsed.group("name")
            shape = parsed.group("shape").replace(" ", "")
            pointer = "*" in parsed.group("prefix") or "&" in parsed.group("prefix")
            slots, note = type_slots(type_name, pointer), ""

            if "constexpr" in qualifiers:
                slots, note = 0, "compile-time"
                initializer = item.split("=", 1)
                if len(initializer) == 2:
                    value = eval_integer(initializer[1], constants)
                    if value is not None:
                        constants[name] = value
            elif "__shared__" in qualifiers:
                slots, note = 0, "shared memory"
            elif slots is not None and shape:
                count = 1
                for dimension in re.findall(r"\[([^\]]*)\]", shape):
                    value = eval_integer(dimension, constants)
                    if value is None:
                        count = None
                        break
                    count *= value
                slots = slots * count if count is not None else None
                indices = re.findall(
                    rf"\b{re.escape(name)}\s*\[([^\]]+)\]", kernel.body
                )
                if any(eval_integer(index, constants) is None for index in indices):
                    note = "dynamic index: may become local memory"

            absolute = kernel.body_offset + match.start()
            line = source.count("\n", 0, absolute) + 1
            variables.append(Variable(line, type_name, name, shape, slots, note))
    return variables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate source-declared 32-bit register slots per CUDA thread."
    )
    parser.add_argument("source", type=Path, help="path to a .cu file")
    parser.add_argument("--kernel", help="only show kernels containing this name")
    args = parser.parse_args()
    if not args.source.is_file() or args.source.suffix != ".cu":
        parser.error(f"expected an existing .cu file: {args.source}")

    source = args.source.read_text()
    kernels = find_kernels(source)
    if args.kernel:
        kernels = [kernel for kernel in kernels if args.kernel in kernel.name]
    if not kernels:
        parser.error("no matching __global__ kernels found")

    for kernel in kernels:
        variables = declarations(kernel, source)
        print(kernel.name)
        for variable in variables:
            slots = "?" if variable.slots is None else str(variable.slots)
            note = f"  [{variable.note}]" if variable.note else ""
            label = f"{variable.type} {variable.name}{variable.shape}"
            print(f"  L{variable.line:<4} {label:<38} {slots:>4} slots{note}")
        known = sum(variable.slots or 0 for variable in variables)
        unknown = sum(variable.slots is None for variable in variables)
        print(f"  declared budget: {known} known 32-bit slots/thread", end="")
        print(f" + {unknown} unknown declaration(s)" if unknown else "")
        print("  compiler temps, argument loads, lifetime reuse, and spills are not included\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
