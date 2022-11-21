# TinyQMIX

This repository is the implementation of "TinyQMIX", which is a cooperative MADRL policy for channel selection in mMTC networks.

<div align=center>
    <img width="700" src="images/model.png" alt="mmtc system model"/>
    <p><strong>Figure 1.</strong> mMTC devices uplink transmission model</p>
</div>

We compare it with different static, tabular Q-learning, and deep Q-learning policies for distributed channel selection methods.

<div align=center>
    <img width="700" src="images/moving_average_delay.png" alt="mean delay"/>
    <p><strong>Figure 2.</strong> Moving average of the access delay. Traffic dynamic changes every 10 seconds
</div>

Over 5 minutes of testing trace, TinyQMIX has the lowest delay, approaching the empirical lower-bound WFLB.

This is the repository for the paper "TinyQMIX: Distributed Access Control for mMTC via Multi-agent Reinforcement Learning" - presented at VTC Fall 2022.

Contact: lethanh@nii.ac.jp