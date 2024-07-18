# L3 Cognitive Architecture Demo 

To set up your machine:
- Install python packages running `pip install -r python_packages.txt`.
- Download, install and abilitate the Unreal Engine [Socketer](https://github.com/How2Compute/Socketer) plugin.

To run the demo application:
- Open the Unreal Engine project `L3_test.uproject` and run the application.
- Launch the python script `L3_CA.py` to connect to the UE application and stream data points to animate character behviour.

## Code Structure
To debug character animation navigate to `L3_test\Content\Characters\Mannequins\Animations` using Content Drawer and open `L3_synch`.
This blueprint connects the character behaviour with Python commands using TCP socket connection, the basic workflow is as follows:
- UE blueprint creates an instance of the socket connection and waits for connection in the event graph.
- By launching `L3_CA.py` a connection is established and the character behaviour is displayed.
- `L3_CA.py` sends data points in the 3D space used as target values for the left hand IK control.
- Such data is stored in a local varible and, it is continously checked by the animgraph to set target location for the left hand IK rig which updates the full body posture.

Useful information:
- The TCP connection consists in the exchange of strings containig 3D end-effector positions and the time interval (delta_t) between each message. The string messages are structured as: `X={end-effector x value} Y={end-effector y value} Z={end-effector z value}; {delta_t value}`.
- The end-effector target positions are computed by python using a discrete time oscillator outputing values relative to character location (origin).
- The IK rig used by the animation blueprint `L3_synch` is located in the folder `L3_test\Content\Characters\Mannequins\Rigs`.
- Connection and message reading happens on a regular basis by the UE animation blueprint so you can kill and launch again the python client to plug the CA to animate character behaviour anytime.
