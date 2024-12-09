cat > snake_game.json <<EOF
{
    "input_files": ["/dev/null"],
    "output_file": "snake_game.py",
    "requirements": "Create a simple Snake game in Python using the pygame library. Include comments explaining the code. The game should have basic functionality with a snake that can move, grow when eating food, and end the game if the snake collides with the walls or itself. Include a scoring system.",
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "force": true
}
EOF

python3 autocoder.py -c snake_game.json
