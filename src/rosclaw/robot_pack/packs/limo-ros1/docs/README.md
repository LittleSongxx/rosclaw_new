# AgileX LIMO ROS 1 Robot Pack

This Pack binds the `limo` e-URDF Body to the independently versioned
`ros-claw/limo-ros-mcp` adapter at commit
`249a26bb950194db71a4451753ec8d504df08607`.

The first REAL capability is `limo.set_initial_pose`. The Agent submits a
validated map-frame estimate to `rosclawd`; the daemon validates an exact
operator permit and starts a fixed-operation ROS Melodic worker. The Agent and
MCP server do not import rospy or publish `/initialpose`.

H1 means only that the signed contract and executor tests pass. H4 requires a
real LIMO, an AMCL subscriber, post-dispatch `/amcl_pose`, a `map -> odom`
transform, a canonical TASK_VERIFIED receipt, and independent review.
