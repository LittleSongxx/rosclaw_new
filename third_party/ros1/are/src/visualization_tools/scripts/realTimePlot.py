#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

import rospy
from std_msgs.msg import Float32, Bool

mpl.rcParams['toolbar'] = 'None'
plt.ion()

time_duration = 0
start_time_duration = 0
first_iteration = 'True'
exploration_finished = False
figure_saved = False

explored_volume = 0;
traveling_distance = 0;
run_time = 0;
max_explored_volume = 0
max_traveling_diatance = 0
max_run_time = 0

time_list1 = np.array([])
time_list2 = np.array([])
time_list3 = np.array([])
run_time_list = np.array([])
explored_volume_list = np.array([])
traveling_distance_list = np.array([])

def timeDurationCallback(msg):
    global time_duration, start_time_duration, first_iteration
    time_duration = msg.data
    if first_iteration == 'True':
        start_time_duration = time_duration
        first_iteration = 'False'

def runTimeCallback(msg):
    global run_time
    run_time = msg.data

def exploredVolumeCallback(msg):
    global explored_volume
    explored_volume = msg.data


def travelingDistanceCallback(msg):
    global traveling_distance
    traveling_distance = msg.data

def explorationFinishCallback(msg):
    global exploration_finished
    if msg.data:
        exploration_finished = True

def save_figure(fig):
    global figure_saved
    if figure_saved:
        return
    
    # 创建保存目录
    save_dir = os.path.expanduser("~/exploration_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"exploration_metrics_{timestamp}.png")
    
    # 保存图片
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    figure_saved = True
    rospy.loginfo(f"Figure saved to: {filename}")
    print(f"\n{'='*60}")
    print(f"探索指标图已保存到: {filename}")
    print(f"{'='*60}\n")

def on_key_press(event):
    global fig
    if event.key == 's':
        save_figure(fig)
        print("手动保存图片完成！")

def listener():
  global time_duration, start_time_duration, explored_volume, traveling_distance, run_time, max_explored_volume, max_traveling_diatance, max_run_time, time_list1, time_list2, time_list3, run_time_list, explored_volume_list, traveling_distance_list, exploration_finished, fig

  rospy.init_node('realTimePlot')
  rospy.Subscriber("/time_duration", Float32, timeDurationCallback)
  rospy.Subscriber("/runtime", Float32, runTimeCallback)
  rospy.Subscriber("/explored_volume", Float32, exploredVolumeCallback)
  rospy.Subscriber("/traveling_distance", Float32, travelingDistanceCallback)
  rospy.Subscriber("/exploration_finish", Bool, explorationFinishCallback)

  fig=plt.figure(figsize=(10,8))
  fig.canvas.mpl_connect('key_press_event', on_key_press)
  
  fig1=fig.add_subplot(311)
  plt.title("Exploration Metrics\n", fontsize=14)
  plt.margins(x=0.001)
  fig1.set_ylabel("Explored\nVolume (m$^3$)", fontsize=11)
  l1, = fig1.plot(time_list2, explored_volume_list, color='r', label='Explored Volume')
  
  fig2=fig.add_subplot(312)
  fig2.set_ylabel("Traveling\nDistance (m)", fontsize=11)
  l2, = fig2.plot(time_list3, traveling_distance_list, color='r', label='Traveling Distance')
  
  fig3=fig.add_subplot(313)
  fig3.set_ylabel("Algorithm\nRuntime (s)", fontsize=11)
  fig3.set_xlabel("Time Duration (s)", fontsize=11)
  l3, = fig3.plot(time_list1, run_time_list, color='r', label='Algorithm Runtime')

  plt.tight_layout()

  count = 0
  r = rospy.Rate(100) # 100hz
  while not rospy.is_shutdown():
      r.sleep()
      count = count + 1

      if count % 25 == 0:
        max_explored_volume = explored_volume
        max_traveling_diatance = traveling_distance
        if run_time > max_run_time:
            max_run_time = run_time

        time_list2 = np.append(time_list2, time_duration)
        explored_volume_list = np.append(explored_volume_list, explored_volume)
        time_list3 = np.append(time_list3, time_duration)
        traveling_distance_list = np.append(traveling_distance_list, traveling_distance)
        time_list1 = np.append(time_list1, time_duration)
        run_time_list = np.append(run_time_list, run_time)

      if count >= 100:
        count = 0
        l1.set_xdata(time_list2)
        l2.set_xdata(time_list3)
        l3.set_xdata(time_list1)
        
        l1.set_ydata(explored_volume_list)
        l2.set_ydata(traveling_distance_list)
        l3.set_ydata(run_time_list)

        fig1.set_ylim(0, max_explored_volume + 500)
        fig1.set_xlim(start_time_duration, time_duration + 10)
        fig2.set_ylim(0, max_traveling_diatance + 20)
        fig2.set_xlim(start_time_duration, time_duration + 10)
        fig3.set_ylim(0, max_run_time + 0.2)
        fig3.set_xlim(start_time_duration, time_duration + 10)

        fig.canvas.draw()
        
        # 探索完成后自动保存
        if exploration_finished:
            save_figure(fig)

if __name__ == '__main__':
  print("\n" + "="*60)
  print("实时探索指标监控启动")
  print("="*60)
  print("功能说明:")
  print("  1. 探索完成后自动保存图片到 ~/exploration_results/")
  print("  2. 按 's' 键可随时手动保存当前图片")
  print("  3. 图片分辨率: 300 DPI")
  print("="*60 + "\n")
  try:
    listener()
  except rospy.exceptions.ROSInterruptException:
    pass
