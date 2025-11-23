Execute python .\Main.py to run.

Choose mode:
1) Player vs NN
2) NN vs NN
Enter 1 or 2: 2
Use MCTS for NN moves? (y/n): y
How many games to play? (default 5): 30

Work in progress

***

https://en.wikipedia.org/wiki/Deep_reinforcement_learning

***

Gomoku (also called Five in a Row) is a two‑player abstract strategy game where the goal is to be the first to align five stones in a row—horizontally, vertically, or diagonally—on a grid board. Black always plays first, players alternate turns, and once placed, stones cannot be moved or removed.

🔹 Core Rules of Gomoku
    • Players: 2 (Black and White). 
    • Board: Standard size is 15×15 intersections (though 19×19 Go boards or smaller variants are sometimes used). 
    • Pieces: Black and White stones (traditionally Go pieces). 
    • Starting move: Black always plays first. 
    • Turn order: Players alternate placing one stone of their color on any empty intersection. 
    • Winning condition: The first player to form an unbroken line of exactly five stones of their color wins. 
        ◦ Lines can be horizontal, vertical, or diagonal. 
        ◦ In some rule sets, six or more stones in a row (an overline) does not count as a win. 

🔹 Variants
    • Free Gomoku (casual play): Any five in a row wins, including overlines. 
    • Renju (tournament rules): To balance Black’s first‑move advantage, Black is forbidden from certain patterns: 
        ◦ Overline: More than five in a row. 
        ◦ Double three: Creating two open rows of three simultaneously. 
        ◦ Double four: Creating two open rows of four simultaneously. 
    • These restrictions make competitive play deeper and fairer. 

🔹 Strategy Basics
    • Attack: Aim to build multiple threats (e.g., two possible lines of five). 
    • Defense: Block your opponent’s attempts to form four in a row. 
    • Center control: Stones near the center give more options for branching lines. 
    • Patterns: Recognize strong threats like an open four (four stones with both ends open), which forces the opponent to block immediately. 

🔹 Summary
    • Objective: Align five stones in a row. 
    • Setup: 15×15 board, Black moves first. 
    • Gameplay: Alternate placing stones; no removals. 
    • Victory: First to five in a row wins (with Renju adding restrictions). 
    • Depth: Simple rules but highly strategic, with traps, forced moves, and long‑term planning. 
	
: Wikipedia – Gomoku
: BitFlap – Complete Gomoku Rules Guide
: Gomoku.com – Rules and Strategy Guide	