import pymysql
import traceback
def creatDB():
    conn=pymysql.connect(host = '127.0.0.1' # 连接名称，默认127.0.0.1
    ,user = 'root' # 用户名
    ,passwd='123456' # 密码
    ,port= 3306 # 端口，默认为3306
    ,charset='utf8' # 字符编码
    )
    cur = conn.cursor() # 生成游标对象
    try:
    # 创建数据库
        DB_NAME = 'nlp'
        cur.execute('DROP DATABASE IF EXISTS %s' %DB_NAME)
        cur.execute('CREATE DATABASE IF NOT EXISTS %s' %DB_NAME)
        conn.select_db(DB_NAME)
        #创建表
        TABLE_NAME = 'admin'
        cur.execute("CREATE TABLE %s(user varchar(100) primary key,password varchar(100))" %TABLE_NAME)

    except:
        traceback.print_exc()
        # 发生错误时会滚
        conn.rollback()
    finally:
        cur.close() # 关闭游标
        conn.close() # 关闭连接
#有关database,只需要执行一次就可以
if __name__=='__main__':
    creatDB()