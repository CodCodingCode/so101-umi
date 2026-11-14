# SO-101 leader arm → kinesthetic-teaching gripper

Turns an SO-101 **leader** arm into an arm you guide by hand with a real gripper and wrist
camera. You move the arm yourself, record its joint angles and wrist video, then replay
them on the follower. The leader's lower-ratio servos are easy to backdrive by hand; the
follower's are not.

These are the SO-101's own follower-gripper parts (official geometry, unchanged), with one
finger loop fused under each jaw. Thumb and index go through the loops; squeeze to close,
spread to open.

## Print (3 parts)

| File                               | Replaces                                                              | Notes                                      |
| ---------------------------------- | --------------------------------------------------------------------- | ------------------------------------------ |
| `soumi_fixed_jaw_loop.stl`         | `Wrist_Roll_Follower_SO101` (the follower's gripper body + fixed jaw) | Loop fused under the fixed jaw             |
| `soumi_moving_jaw_loop.stl`        | `Moving_Jaw_SO101`                                                    | Loop fused under the moving jaw            |
| `SO-ARM101_camera_wrist_mount.stl` | n/a                                                                   | Same wrist-camera mount as on the follower |

PETG or PLA+, 5 walls, 50% infill. Strongest with the loop's hole pointing straight up (part
standing on end), so each layer is a full ring. Tree supports. All files are in mm.

You don't need the handle or the fit-test coupon from the main folder for this.

## Assembly

1. Take the leader's handle and trigger off the wrist-roll servo.
2. Bolt `soumi_fixed_jaw_loop` onto the wrist-roll servo horn exactly as the follower's gripper
   body goes on (same screws, same hole pattern).
3. Move the leader's 6th servo (the trigger servo, usually the easy-to-turn 1/147 version; check
   the label) into the gripper cage, the same way as on the follower.
4. Mount `soumi_moving_jaw_loop` on that servo's horn, at the same horn position as on your follower.
5. Add the camera mount and camera exactly as on the follower.
6. Recalibrate this arm. The gripper joint now has the follower gripper's range, not the trigger's.

## Using it

- Turn torque off on all 6 servos. Record joint positions and the wrist camera at 30–50 Hz,
  then replay the positions on the follower.
- Fingers in the loops open and close the jaws. Use your other hand or palm on the wrist for big moves.
- The wrist camera never sees the loops or your fingers in them (checked in CAD against the
  measured lens).

## Collision check (CAD, `../src/arm_sweep.py`)

- I swept 7,605 arm poses: shoulder, elbow, wrist bend, wrist roll and jaw angle.
- The loops add **no** collisions with the wrist or forearm anywhere in the joint ranges.
- They add a collision in 37 poses, all with the shoulder bracket or upper arm. In each one
  the gripper is working within ~10–17 cm of the robot's own base, the jaws point down or
  back toward the base, and the wrist is rolled so the loops face it. If you work that
  close to the base, roll the wrist the other way.
- The loops stick out ~56 mm from the gripper's centre line, less than the stock wrist
  camera (~88 mm).

## Known limitation: grip force

Rigid jaws record position only. Touching an object and squeezing it give the same reading,
so a policy trained on this data closes until contact and never squeezes. Fixes:

- Soft (fin-ray) fingertips, so squeeze shows up as extra closing and in the image.
- Relabel the data: when the jaw stops short of closed, set the target a few mm tighter.
