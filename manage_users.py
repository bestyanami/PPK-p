import json
import os
import getpass
from werkzeug.security import generate_password_hash
import argparse

USERS_FILE = 'users.json'

def load_users():
    """加载用户文件，如果不存在则返回空字典"""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_users(users):
    """保存用户数据到文件"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def add_user(username, password):
    """添加新用户或更新现有用户密码"""
    users = load_users()
    if username in users:
        print(f"用户 '{username}' 已存在。是否要更新其密码? (y/n): ", end='')
        if input().lower() != 'y':
            print("操作已取消。")
            return

    password_hash = generate_password_hash(password)
    users[username] = {'password_hash': password_hash}
    save_users(users)
    print(f"用户 '{username}' 已成功添加/更新。")

def delete_user(username):
    """删除指定用户"""
    users = load_users()
    if username not in users:
        print(f"错误: 用户 '{username}' 不存在。")
        return
    
    del users[username]
    save_users(users)
    print(f"用户 '{username}' 已被删除。")

def list_users():
    """列出所有已存在的用户"""
    users = load_users()
    if not users:
        print("系统中没有用户。")
        return
    
    print("系统中的用户列表:")
    for username in users:
        print(f"- {username}")

def main():
    parser = argparse.ArgumentParser(description="用户管理工具")
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的命令')

    # 添加用户命令
    parser_add = subparsers.add_parser('add', help='添加一个新用户或更新密码')
    parser_add.add_argument('username', type=str, help='要添加的用户名')

    # 删除用户命令
    parser_del = subparsers.add_parser('delete', help='删除一个用户')
    parser_del.add_argument('username', type=str, help='要删除的用户名')

    # 列出用户命令
    subparsers.add_parser('list', help='列出所有用户')

    args = parser.parse_args()

    if args.command == 'add':
        password = getpass.getpass(f"请输入用户 '{args.username}' 的密码: ")
        password_confirm = getpass.getpass("请再次输入密码以确认: ")
        if password != password_confirm:
            print("两次输入的密码不匹配。操作已取消。")
            return
        add_user(args.username, password)
    elif args.command == 'delete':
        delete_user(args.username)
    elif args.command == 'list':
        list_users()

if __name__ == '__main__':
    main()