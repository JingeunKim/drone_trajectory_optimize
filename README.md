# MAC: Memetic Actor-Critic Framework for UAV Path Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

OFFICIAL IMPLEMENTATION of the paper: **"MAC: Memetic Actor-Critic Framework for UAV Path Planning"** (Submitted to GECCO '26).
##  Abstract
Unmanned Aerial Vehicles (UAVs) play a pivotal role in maritime Search and Rescue (SAR) missions, yet generating optimal flight paths remains a significant challenge. Traditional meta-heuristic algorithms, such as genetic algorithms (GA), often suffer from premature convergence to local optima, while reinforcement learning (RL) approaches typically struggle with low sample efficiency and slow initial convergence. To address these limitations, this paper proposes a hybrid framework termed Memetic Actor-Critic (MAC). The MAC framework integrates the decision-making policy of Actor-Critic with the global search capabilities of a memetic algorithm. Specifically, we introduce a task-specific local refinement mechanism that utilizes state-value estimations to refine the actor's policy, enabling precise exploitation of high-value regions beyond immediate rewards. The proposed method was thoroughly evaluated using a realistic SAR scenario constructed from oceanographic particle simulation data collected from the East Sea of South Korea. Experimental results demonstrate that MAC significantly outperforms standalone GA, MA, AC, and other variants in terms of both solution quality and convergence stability, proving its effectiveness in balancing global exploration and local exploitation.

###  Experimental Results (5 Instances)

Here are the trajectory visualization results for all 5 instances. MAC consistently navigates to high-density regions effectively.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/final_visualization_outside_i1.png" width="100%">
      <br><b>Instance 1</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/final_visualization_outside_i2.png" width="100%">
      <br><b>Instance 2</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/final_visualization_outside_i3.png" width="100%">
      <br><b>Instance 3</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/final_visualization_outside_i4.png" width="100%">
      <br><b>Instance 4</b>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="assets/final_visualization_outside_i5.png" width="50%">
      <br><b>Instance 5</b>
    </td>
  </tr>
</table>
