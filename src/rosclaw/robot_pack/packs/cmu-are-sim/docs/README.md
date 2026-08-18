# CMU ARE simulation Robot Pack

`cmu-are-sim` binds the ROS1 Noetic CMU Autonomous Exploration Environment and
ARiADNE2 stack to the ROSClaw daemon through a local rosbridge WebSocket.

The pack exposes only `SHADOW` capabilities.  The daemon owns the rosbridge
connection and publishes the fixed CMU topics; Agents cannot select arbitrary
topics, device paths, drivers, or `/cmd_vel`.

Large checkpoints, planner paths, and Gazebo meshes are supplied out of band.
See `docs/assets/cmu-are-assets.yaml` and the Docker Compose file for the
read-only mount contract.
