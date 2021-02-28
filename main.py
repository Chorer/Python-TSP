import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QMessageBox
from PyQt5.QtCore import QFile
import pso_gui
from pso import PSO


# 全局变量
data = ''
iter_max = 0
num_group = 0
log_str = ''

# 读取数据
def readFile():
  global data
  fname = QFileDialog.getOpenFileName(MainWindow, "Open File", "./", "Txt (*.txt)")
  path = fname[0]
  if path:
    data = processData(path)
    QMessageBox.information(MainWindow,'导入数据文件','导入成功！')
    ui_obj.textEdit_3.setText('文件路径：' + '\n' + path)
# 处理数据
def processData(path):
  lines = open(path, 'r').readlines()
  assert 'NODE_COORD_SECTION\n' in lines
  index = lines.index('NODE_COORD_SECTION\n')
  data = lines[index + 1:-1]
  tmp = []
  for line in data:
      line = line.strip().split(' ')
      if line[0] == 'EOF':
          continue
      tmpline = []
      for x in line:
          if x == '':
              continue
          else:
              tmpline.append(float(x))
      if tmpline == []:
          continue
      tmp.append(tmpline)
  data = tmp
  return np.array(data)[:,1:]

# 绘制城市坐标分布图
def drawCitiesPoint():
  plt.figure(figsize=(60,30))
  plt.subplots_adjust(hspace = 0.4)
  plt.suptitle('仿真实验结果')
  plt.subplot(2, 2, 1)
  plt.title('城市坐标分布图')
  plt.xlabel('km')
  plt.ylabel('km')
  plt.plot(data[:, 0], data[:, 1],'s',markerfacecolor = '#18e225',markeredgecolor = 'black')

def runTest():
  global iter_max
  global num_group
  if ui_obj.lineEdit.text() and ui_obj.lineEdit_2.text():
    iter_max = int(ui_obj.lineEdit.text())
    num_group = int(ui_obj.lineEdit_2.text())
  else:
    iter_max = 100
    num_group = 150
  # 绘制图一
  drawCitiesPoint()
  # 运行主程序进行仿真测试
  pso = PSO(num_city = data.shape[0],data = data.copy(),num_group = num_group,iter_max = iter_max,ui_obj = ui_obj)
  Best_path, Best_length = pso.run()
  Best_path = list(map(lambda x: x+1,Best_path))
  frontText = ui_obj.textEdit.toPlainText()
  backText = '迭代结束后的最短距离是：' + str(Best_length) + '\n' + '迭代结束后的最佳路径是：' + '\n' + str(Best_path)
  ui_obj.textEdit.setText(frontText + backText)    
  plt.show()

# 主程序
if __name__ == '__main__':
  # UI 初始化
  app = QApplication(sys.argv)
  MainWindow = QMainWindow()
  ui_obj = pso_gui.Ui_MainWindow()
  ui_obj.setupUi(MainWindow)

  # 事件绑定
  ui_obj.btn_readFile.clicked.connect(readFile)
  ui_obj.btn_runTest.clicked.connect(runTest)

  # UI 展示
  MainWindow.show()
  sys.exit(app.exec_())