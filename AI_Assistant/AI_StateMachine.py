from enum import Enum, unique

@unique
class States(Enum):
    CHAT = 1
    LEARN = 2
    QUIT = 3

