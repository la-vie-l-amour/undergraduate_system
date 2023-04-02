from PySide2.QtGui import QPixmap
import pymysql
from lib.share import shareInf
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QGraphicsScene
import resource.resource_ui   #不要删，它必须引进来，否则会导致图片显示不出来

from recognise import regcognise

class Win_Login:
    def __init__(self):

        self.ui = QUiLoader().load("./ui/login.ui")
        self.ui.button_login.clicked.connect(self.onSignIn)
        self.ui.sign_login.clicked.connect(self.onSignUp)
        self.ui.edit_password.returnPressed.connect(self.onSignIn)
        self.ui.sign_password.returnPressed.connect(self.onSignUp)
        self.ui.widget_3.hide()
        self.ui.login.clicked.connect(self.change_widget2)
        self.ui.sign.clicked.connect(self.change_widget3)

    def onSignUp(self):
        password = self.ui.sign_password.text().strip()
        username = self.ui.sign_username.text().strip()
        conn = pymysql.connect(host='127.0.0.1'  # 连接名称，默认127.0.0.1
                                       , user='root'  # 用户名
                                       , passwd='123456'  # 密码
                                       , port=3306  # 端口，默认为3306
                                       , db='nlp'  # 数据库名称
                                       , charset='utf8'  # 字符编码
                                       )
        sql = "INSERT INTO `admin` (`user`,`password`) VALUES ('%s','%s')"%(username,password)
        try:
            with conn.cursor() as cur:
                cur.execute(sql)  # 执行插入的sql语句
            conn.commit()  # 提交到数据库执行
        except:
            conn.rollback()  # 如果发生错误则回滚
            QMessageBox.warning(self.ui, '注册失败', '账号已存在')
            conn.close()  # 关闭数据库连接
            return
        conn.close()  # 关闭数据库连接
        QMessageBox.warning(self.ui, '注册成功', '请登录')
        return


    def onSignIn(self):
        password = self.ui.edit_password.text().strip()
        username = self.ui.edit_username.text().strip()
        isuser = False
        conn = pymysql.connect(host='127.0.0.1'  # 连接名称，默认127.0.0.1
                                       , user='root'  # 用户名
                                       , passwd='123456'  # 密码
                                       , port=3306  # 端口，默认为3306
                                       , db='nlp'  # 数据库名称
                                       , charset='utf8'  # 字符编码
                                       )
        cur = conn.cursor()  # 生成游标对象
        sql = "select * from `admin` "  # SQL语句
        cur.execute(sql)  # 执行SQL语句
        data = cur.fetchall()  # 通过fetchall方法获得数据
        cur.close()  # 关闭游标
        conn.close()  # 关闭连接
        for line in data:
            if username == line[0]:
                isuser = True
                trpass = line[1]
        if isuser == False:
            QMessageBox.warning(self.ui, '登录失败', '账号错误')
            return
        if password != trpass:
            QMessageBox.warning(self.ui, '登录失败', '密码错误')
            return

        shareInf.menuWin = Win_menu()
        shareInf.menuWin.ui.show()  # 跳转
        self.ui.edit_password.setText('')
        self.ui.hide()  # 关闭自身

    def change_widget3(self):
        self.ui.widget_2.hide()
        self.ui.widget_3.show()

    def change_widget2(self):
        self.ui.widget_3.hide()
        self.ui.widget_2.show()

class Win_menu:
    def __init__(self):
        self.ui = QUiLoader().load("./ui/menu.ui")
        #登出
        self.ui.keyfile.clicked.connect(self.handleCalc_picture)
        self.ui.recognize.clicked.connect(self.handleCalc_recognize)


        self.ui.textBrowser_2.hide()
        self.ui.textBrowser_3.hide()
        self.ui.pushButton_2.clicked.connect(self.changeBrower2)
        self.ui.pushButton_3.clicked.connect(self.changeBrower3)
        self.ui.pushButton_4.clicked.connect(self.changeBrower4)


    def handleCalc_picture(self):
        self.filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要抽取的图片",  # 标题
            r"C://",  # 起始目录
            "Images (*.png *.jpg)"  # 选择类型过滤项，过滤内容在括号中
        )
        if self.filePath == "":
            QMessageBox.warning(self.ui, '选择失败', '请重新选择图片')
            return

        scene = QGraphicsScene()
        scene.addPixmap(QPixmap(self.filePath).scaled(self.ui.graph.size()))
        self.ui.graph.setScene(scene)
        self.ui.graph.show()


    def handleCalc_recognize(self):
        lpn = regcognise(self.filePath)

        QMessageBox.about(self.ui,
                          '车牌号',
                          f'''车牌号为：\n{lpn}''')


    def changeBrower2(self):
        self.ui.textBrowser_2.hide()
        self.ui.textBrowser_3.hide()
        self.ui.textBrowser.show()
    def changeBrower3(self):
        self.ui.textBrowser_2.show()
        self.ui.textBrowser_3.hide()
        self.ui.textBrowser.hide()
    def changeBrower4(self):
        self.ui.textBrowser_2.hide()
        self.ui.textBrowser_3.show()
        self.ui.textBrowser.hide()
if __name__=='__main__':
    app = QApplication([])
    QApplication.addLibraryPath('./plugins')
    shareInf.loginWin = Win_Login()
    shareInf.loginWin.ui.show()
    app.exec_()