if [ "x$2" == "x" ]; then 
  echo usage: $0 key value
  exit 1
fi

key="$1"
value="$2"

echo "select * from control_settings where setting = '$1'" | sqlite3 irc_bot_log.db
echo "update control_settings set value='$2' where setting = '$1'"


echo "update control_settings set value='$2' where setting = '$1'" | sqlite3 irc_bot_log.db 
echo "select * from control_settings where setting = '$1'" | sqlite3 irc_bot_log.db

