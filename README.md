(Link to the older manual version): https://github.com/yvsharma02/SubwaySurfersAI <br/>


How to start the devcontainer:

Things to have on host machine:
1) Docker
2) VSCode
3) DevContainer Extension for VSCode
4) NVidia Drivers ALONG with Nvidia Container toolkit.
5) An image of an android emulator needs to be located at data/avd/avd.zip. This needs to be frozen at a state where the game Subway surfers is already running, the tutorials are completed, and atleast one game is played after launch. (Esentially, the "Play" button should be on screen.). Positions are some settings might need to be changed depending on the emulator resolution. Contact me if you need the image I used. <br/> <br/>

VSCode should automatically give a prompt to open the project inside a devcontainer when all 3 steps are met.
If you want to use Git while being inside the container, the following things also need to be installed:

4) Git
5) GitCredentailManager (Make sure to proprely set this up. Good luck with that :) )

To run the project (from inside dev-container): Run: ./setup/launch.sh

Sample Run:

https://github.com/user-attachments/assets/544474dc-55cc-472e-8016-1a5e8a681b17

This works be starting with an uninitialized model, predicting which actions it should eliminate.
The action with the least confidence in elimination it performed (after considering some cooldowns and stuff).
If the taken action leads to the game ending, that action is marked as eliminated, and saved as a data point.
After few such tries, a new model is training on the data which contains the data of the eliminiated actions for each frame we just created.

Here are the overall results:

<img width="1000" height="800" alt="graph" src="https://github.com/user-attachments/assets/099230a0-fbac-495e-a4a5-0d2e50a4ed2c" />

Network Architecture:

<img width="512" height="768" alt="architecture" src="https://github.com/user-attachments/assets/84a6558c-4354-47fe-80e6-061c983a1685" />
