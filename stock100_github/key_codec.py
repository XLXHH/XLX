"""Reversible key encoding, NOT encryption or a replacement for secret storage."""
import base64
import binascii
import re


class KeyCodecError(ValueError):
    pass


def _valid_key(key):
    if not isinstance(key, str) or not 3 <= len(key) <= 512 or not re.fullmatch(r"[\x21-\x7e]+", key):
        raise KeyCodecError("密钥应为3～512个可打印英文字符，不能含空格或换行。")
    return key


def encode_key(key):
    key = _valid_key(key.strip())
    # Packing lowercase hex rather than encoding its ASCII spelling saves space.
    match = re.fullmatch(r"sk-([0-9a-f]+)", key)
    if match and len(match[1]) % 2 == 0:
        content = bytes.fromhex(match[1])
        prefix = "D1."
    else:
        content = key.encode("ascii")
        prefix = "K1."
    return prefix + base64.urlsafe_b64encode(content).decode("ascii").rstrip("=")


def decode_key(encoded):
    if not isinstance(encoded, str):
        raise KeyCodecError("请输入编码工具生成的密钥。")
    encoded = encoded.strip()
    if len(encoded) > 1024 or not re.fullmatch(r"(?:D1|K1)\.[A-Za-z0-9_-]+", encoded):
        raise KeyCodecError("编码格式无效，请使用编码密钥工具生成以 D1. 或 K1. 开头的内容。")
    prefix, payload = encoded.split(".", 1)
    try:
        content = base64.b64decode(payload + "="*((-len(payload)) % 4), altchars=b"-_", validate=True)
        # Reject ambiguous/noncanonical encodings rather than silently change input.
        if base64.urlsafe_b64encode(content).decode("ascii").rstrip("=") != payload:
            raise ValueError()
        key = "sk-" + content.hex() if prefix == "D1" else content.decode("ascii")
        return _valid_key(key)
    except (ValueError, UnicodeError, binascii.Error):
        raise KeyCodecError("编码内容损坏或不完整，请重新编码后粘贴。") from None


def main():
    import getpass
    print("密钥编码工具：只在本机转换，不联网、不写入文件。编码不等于加密。")
    key = getpass.getpass("请粘贴原始 API Key（输入不显示），然后回车：")
    try:
        encoded = encode_key(key)
    except KeyCodecError as exc:
        print(str(exc))
        return 1
    print("\n请复制下面一行，粘贴到看板的“编码后的 DeepSeek 密钥”：\n")
    print(encoded)
    print(f"\n长度：原始{len(key.strip())}字符 → 编码后{len(encoded)}字符。请按密钥保管，不要上传GitHub。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
