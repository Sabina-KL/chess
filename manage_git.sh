#!/bin/bash

if [ "$1" == "heroku" ]; then
  # Remove LFS tracking for Heroku
  git lfs untrack "chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git lfs migrate export --include="chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git rm --cached "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Remove file from Git index but keep locally
  git rm .gitattributes  # Remove the LFS tracking from .gitattributes
  git commit -m "Remove LFS tracking for Heroku"
  git push heroku main
elif [ "$1" == "clear history" ]; then
  # Clear history and remove file from LFS tracking
  echo "Clearing history and untracking CIFAR-10 file..."
  git lfs untrack "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Untrack from LFS
  git rm --cached "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Remove the file from git history
  git lfs migrate import --include="chess-service/src/assets/templates/cifar-10-python.tar.gz" # Migrate file to regular Git
  git commit -m "Remove CIFAR-10 file from history and clear LFS tracking"
  git push origin main
elif [ "$1" == "track" ]; then
  # Track the file with LFS and commit
  echo "Tracking CIFAR-10 file with LFS..."
  git lfs track "chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git add .gitattributes  # Add .gitattributes to staging
  git add "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Add the LFS-tracked file to staging
  git commit -m "Track CIFAR-10 file with LFS"
  git push origin main
else
  # Default action, if no known parameter is passed
  echo "Invalid or missing parameter. Please use 'heroku', 'clear history', or 'track'."
fi

# Usage examples:
# ./manage_git.sh github         -> To push to GitHub with LFS tracking
# ./manage_git.sh heroku         -> To push to Heroku without LFS tracking
# ./manage_git.sh "clear history" -> To clear the history and remove LFS tracking
# ./manage_git.sh track          -> To re-add the file with LFS tracking
