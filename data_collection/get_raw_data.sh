echo "Downloading Replication dataset"
FILE=debian_data.csv
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    # https://drive.google.com/open?id=1W0hD9CuqY4V5QzSEyjLhOHLlYHWjKDkp
    fileid="1IE4qwQj9BkyeMD4-7RvrepocujHsYz9D"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    rm ./cookie
fi
