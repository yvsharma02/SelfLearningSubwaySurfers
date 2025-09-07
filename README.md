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

dev-cache contains optional cache so that I don't have to wait 15 minutes for everytime to be downloaded everytime. To use
1) apt-cache abd pip cache: Just run docker-compose up in that folder. If compose-up does not run, no cache will be use for pip and apt.
2) android-studio cache: just install required android sdk files inside dev-cache/generated/android-sdk. This can be found inside workdir/android-sdk after the first build, or you can download from here. https://dl.google.com/android/repository/cmdline-tools-linux-11076708_latest.zip Make sure the final structure is like: dev-cache/generated/android-sdk/cmdline-tools