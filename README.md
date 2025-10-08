How to start the devcontainer:

Things to have on host machine:
1) Docker
2) VSCode
3) DevContainer Extension for VSCode
4) NVidia Drivers ALONG with Nvidia Container toolkit.

VSCode should automatically give a promt to open the project inside a devcontainer when all 3 steps are met.
If you want to use Git while being inside the container, the following things also need to be installed:

4) Git
5) GitCredentailManager (Make sure to proprely set this up. Good luck with that :) )

To run the project (from inside dev-container): Run: ./setup/launch.sh