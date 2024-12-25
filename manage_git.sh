#!/bin/bash

if [ "$1" == "heroku" ]; then
  # Remove LFS tracking for Heroku (since Heroku can't handle LFS files)
  git lfs untrack "chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git lfs migrate export --include="chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git rm --cached "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Remove file from Git index but keep locally
  git rm .gitattributes  # Remove the LFS tracking from .gitattributes
  git commit -m "Remove LFS tracking for Heroku" 
  git push heroku main
else
  # Ensure the file is tracked with LFS for GitHub
  git lfs track "chess-service/src/assets/templates/cifar-10-python.tar.gz"
  git add .gitattributes  # Add .gitattributes to staging
  git add "chess-service/src/assets/templates/cifar-10-python.tar.gz"  # Add the LFS-tracked file to staging
  git commit -m "Restore LFS tracking for GitHub"
  git push origin main
fi

# Usage examples:
# ./manage_git.sh github  -> To push to GitHub with LFS tracking
# ./manage_git.sh heroku  -> To push to Heroku without LFS tracking
