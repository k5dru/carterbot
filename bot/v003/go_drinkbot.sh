#
# optional: import our Python environment. 
. ~/2025/bin/activate 

# kill stragglers. This is a bit, anti-surgical. 
killall python3 

# if there isn't a history database, make it: 
if [ ! -e irc_bot_log.db ]; then 
    python3 bot_db.py  # if this is called with no parameters instead of imported it will create the db
fi

if [ ! -e irc_bot_log.db -o -z irc_bot_log.db ]; then 
    echo something is wrong - there should be an irc_bot_log.db created by now. 
    exit 1
fi

# clear straggler requests
echo "delete from request_response where posted_time is null;" | sqlite3 irc_bot_log.db 

# set our nick. Personality will be read from $nick.txt file
nick=drinkbot

# choose which system message to include in the prompt
ln -fs system_message_medium.txt system_message.txt

# ./botset.sh model_large deepseek/deepseek-v3.1-terminus
# also tried  nvidia/llama-3.1-nemotron-ultra-253b-v1:free but worse
./botset.sh model_large z-ai/glm-4.6
./botset.sh personality $nick
./botset.sh response_factor_window 120   # was 120
./botset.sh response_factor 1.2          # was 1.5
./botset.sh memory 80  # was 45        
./botset.sh max_line_length 400   # was 255
./botset.sh safety True           # 

echo "select * from control_settings" | sqlite3 irc_bot_log.db 

chan="#botparty"

# start the IRC frontend in the background
python3 bot_irc.py -c "$chan" -n $nick & 
sleep 3

# start the API backend in the foreground
# for local API:  
# python3 bot_api.py --api-provider local -n $nick 
# for hyperbolic (HYPERBOLIC_API_KEY must be set)
# python3 bot_api.py --api-provider hyperbolic -n $nick 
# for openrouter (OPENROUTER_API_KEY must be set)
python3 bot_api.py --api-provider openrouter -n $nick 

