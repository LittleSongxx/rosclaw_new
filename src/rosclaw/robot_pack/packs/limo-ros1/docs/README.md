# AgileX LIMO ROS 1 Robot Pack

This Pack binds the `limo` e-URDF Body to the independently versioned
`ros-claw/limo-ros-mcp` adapter at commit
`8242a33991187c62020000f8faf51dc44f0f9d44` (MCP 0.7.0).

The first REAL capability is `limo.set_initial_pose`. The Agent submits a
validated map-frame estimate to `rosclawd`; the daemon validates an exact
operator permit and starts a fixed-operation ROS Melodic worker. The Agent and
MCP server do not import rospy or publish `/initialpose`.

H1 means only that the signed contract and executor tests pass. H4 requires a
real LIMO, an AMCL subscriber, post-dispatch `/amcl_pose`, a `map -> odom`
transform, a canonical TASK_VERIFIED receipt, and independent review.

Revision 0.1.2 waits for a converged AMCL sample within the bounded
verification window, so an immediate transient `/amcl_pose` sample does not
produce a false-negative receipt.

Revision 0.1.3 adds in-context MCP operator confirmation for REAL initial-pose
requests and allows five seconds for read-only ROS CLI startup on LIMO ARM
hosts. Clients without MCP form elicitation now fail immediately instead of
waiting for a confirmation response they cannot render. Permit material remains
internal to the trusted ROSClaw host boundary.

Revision 0.1.4 matches the Dabai U3 streams launched by
`astra_camera/dabai_u3.launch`: `/camera/color/image_raw`,
`/camera/depth/image_raw`, and `/camera/depth/points`. The MCP returns bounded
metadata summaries and never exposes the raw image or point-cloud arrays.
ROS CLI array placeholders are decoded into their true bounded byte and field
counts, rather than reporting the placeholder string length.

Revision 0.1.5 locks the adapter revision that expands read-only inspection to
29 MCP tools and 27 ROS observations. It adds bounded Dabai device and
color/depth/IR camera-state summaries, host audio playback/capture inventory,
in-memory microphone level measurement, display/touch inventory, USB peripheral
inventory, and platform health. Microphone samples are discarded immediately:
no recording or raw audio content is retained or returned. IR endpoints may be
present but are reported inactive when the driver publishes no frames. Front
OLED and chassis RGB lights remain declared but unbound until a stable host or
ROS interface is available.
