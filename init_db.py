import sqlite3

# 数据库名称（你也可以改为绝对路径）
db_path = "/home/featurize/work/Image_Website/static/user_auth.db"

# 连接数据库（如果没有就创建）
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建 users 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("✅ 用户数据库已初始化并创建 users 表")
