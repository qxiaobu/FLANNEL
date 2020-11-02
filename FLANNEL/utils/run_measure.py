from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import csv
#from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve as roc_curve
from sklearn.metrics import average_precision_score as auc
class MeasureR(object):
    def __init__(self, fdir, loss, acc):
      self.fdir = fdir
      self.file_path = os.path.join(self.fdir, 'result_detail.csv')
      self.wfile_path = os.path.join(self.fdir, 'measure_detail.csv')
      
      self.acc = acc
      self.loss = loss
      print (self.fdir)
      print (self.file_path)
      print (self.acc, self.loss)
      
    def output(self):
      with open(self.file_path, 'r') as f:
        csv_reader = csv.reader(f)
        p0 = []
        p1 = []
        p2 = []
        p3 = []
        l0 = []
        l1 = []
        l2 = []
        l3 = []
        
        target_s = np.zeros(4).astype(float)
        predict_s = np.zeros(4).astype(float)
        tp_s = np.zeros(4)
        for row in csv_reader:
          pv = np.array(row[:4])
          rv = np.array(row[4:])
          p_id = np.argmax(pv)
          t_id = np.argmax(rv)
          target_s[t_id] += 1.
          predict_s[p_id] += 1.
          if t_id == p_id:
            tp_s[t_id] += 1.
          p0.append(float(row[0]))
          p1.append(float(row[1]))
          p2.append(float(row[2]))
          p3.append(float(row[3]))
          l0.append(int(float(row[4])))
          l1.append(int(float(row[5])))
          l2.append(int(float(row[6])))
          l3.append(int(float(row[7])))
        p0 = np.array(p0)  
        p1 = np.array(p1)
        p2 = np.array(p2)  
        p3 = np.array(p3)
        l0 = np.array(l0)
        l1 = np.array(l1)
        l2 = np.array(l2)
        l3 = np.array(l3)

        precision = tp_s/predict_s
        recall = tp_s/target_s 
        with open(self.wfile_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Type','Covid-19','Pneumonia Virus','Pneumonia Bacteria','Normal'])            
            csv_writer.writerow(['Precision']+list(precision))
            csv_writer.writerow(['Recall']+list(recall))
        fpr_0,tpr_0,threshold_0=roc_curve(l0,p0)
        roc_auc_0=auc(l0,p0)
        fpr_1,tpr_1,threshold_1=roc_curve(l1,p1)
        roc_auc_1=auc(l1,p1)
        fpr_2,tpr_2,threshold_2=roc_curve(l2,p2)
        roc_auc_2=auc(l2,p2)
        fpr_3,tpr_3,threshold_3=roc_curve(l3,p3)
        roc_auc_3=auc(l3,p3)
        plt.figure(figsize=(10,10))
        plt.plot(fpr_0, tpr_0, color='red',
        lw=2, label='Covid-19 (AP = %0.4f)' % roc_auc_0) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr_1, tpr_1, color='black',
        lw=2, label='Pneumonia Virus (AUC = %0.4f)' % roc_auc_1) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr_2, tpr_2, color='orange',
        lw=2, label='Pneumonia Bacteria (AUC = %0.4f)' % roc_auc_2) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr_3, tpr_3, color='blue',
        lw=2, label='Normal (AUC = %0.4f)' % roc_auc_3) ###假正率为横坐标，真正率为纵坐标做曲线

        plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
        plt.text(.6, .5, 'Test Loss: %f, Acc: %f'%(self.loss, self.acc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
#        foo_fig = plt.gcf()
#        saved_path = os.path.join(self.fdir, 'results.png')
#        if os.path.exists(saved_path):
#          os.remove(saved_path)
#        foo_fig.savefig(saved_path)
        plt.show()

if __name__ == '__main__':
    dir1 = './results/fodanetalex_20200326_multiclass_softmax'
    mr1 = MeasureR(dir1, 0.1, 0.1)
    mr1.output()
