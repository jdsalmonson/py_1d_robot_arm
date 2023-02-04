[//]: <> (shift-command-V to toggle preview)
OneD Robot
==========

This is an implementation of a 1D Robot.  It is comprised of only a finger that moves back and forth according to an applied force.  The finger tip interacts with an object; touching, pushing or bouncing off of it.  A force sensor on the finger tip records the touch interaction with the object.  As such this is one of the simplest robot simulations of which one can conceive.  For more description, see my [blog post](https://jdsalmonson.github.io/1D-robot/).

***Example 1:*** a force is discontinuously applied from 1 to 5 seconds that accelerates the finger robot until it contacts the object.  The force is below the threshold of static friction of the object, so both finger and object are rendered static.  A second, stronger force is applied form 6.5 to 7.5 seconds which exceeds the objects static friction and thus both finger and object accelerate together, subject to their respective drag and kinetic friction forces.  Once the applied force terminates, the drag force on the finger is stronger than the kinetic friction of the object, so the finger robot decelerates faster and detaches from the object.
![](/images/oneD_robot.gif)

***Example 2:*** a force is ramped up and down between 0 and 3.2 seconds, accelerating the finger robot.  After the applied force, the finger coasts until it bounces off the object with an elastic coefficient of 0.5 (half of the momentum is absorbed).  A second, stronger force is ramped up and down, turning the finger trajectory around and hitting the object with a force exceeding its static friction, thus accelerating both objects together.
![](/images/oneD_robot2.gif)