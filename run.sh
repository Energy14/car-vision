#!/bin/bash
mamba activate
python /sl/src/streamlink/__main__.py --player-external-http --player-external-http-port 2727 --player-external-http-interface "0.0.0.0" "https://www.youtube.com/watch?v=1fiF7B6VkCk" 720p &
flask run