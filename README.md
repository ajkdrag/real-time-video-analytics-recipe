# realtime-video-analytics-recipe
This repo addresses a simplified use-case of event detection in videos using computer vision and deep learning. This recipe can easily be scaled up and extended to address more complex scenarios. 

<details open>
  <summary style="cursor: pointer;">
          <strong> Overview </strong>
  </summary>
<br>

Let's consider a minimal yet common scenario where one would like to detect certain __events__ from a live feed in __realtime__. Given the complexity of the event, the analytical solutions differ, but a vast majority of scenarios end up requiring ML based solutions, owing to the fact that ML can provide robust solutions that don't break altogether upon slight environmental/scenic changes.

---

#### The Plot
Almost all of us have encountered and hated mosquitoes at least once in our lifespan. Mosquito bats :tennis: are effective at the job they do and we will be tackling the problem of detecting certain toy events with this rather *odd* object :rofl:.

#### Events
- __Swatting mode__ : There's a button on the side of the bat, which when pressed puts the bat into swatting mode and additionally lights a tiny red LED that indicates the same.
- __Blinking mode__ : The user can (for fun), press and release the swat button in quick successions, thereby making the red LED indicator flicker. 
- __Light colors__ : Additional to the red led, there exists a blue led as well which is solely for vision during the dark. This requirement is for detecting the colors of these illuminated LEDs on the bat.

These are really as unrealistic as a toy example can get, but as we shall see, the components needed to tackle these events are applicable to other realistic events.
</details>

<details open>
  <summary style="cursor: pointer;">
          <strong> Solution </strong>
  </summary>
<br>

All the events discussed, require one to locate the illuminated LED on the bat (Object Detection to the rescue). That's a good start, but let's dive deeper into the solutions for each event. 
1. For the 1st event, it suffices to check if the red LED on the bat is illuminated or not. 
2. For the 2nd event, we need to check if the red LED has been *alternating* between on and off states in a certain time duration. 
3. Finally, for the 3rd requirement, all we need to do is detect the colored LEDs separately. We can label the classes separately for each color and use the OD model to give us the colors, but any new addition or changes in the LED color would require retraining. A better approach would be to use OD just to get the illuminated LEDs and then use CV for identifying the colors of these regions. Easy!

We can now list the steps needed as follows:
- Dataset preparation
   - Data creation
   - Data annotation
   - Data curation
   - Data augmentation (if needed)
- Training
   - Model selection
   - Baseline
   - Adjustments & Tuning
- Versioning
   - Model evaluation
   - Artifacts storage
- Deployment
   - Serving framework selection
   - Latency check
- Inference
   - Pre and post processing
   - End-to-End tests