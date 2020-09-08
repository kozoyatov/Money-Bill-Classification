import enum


class Type(enum.Enum):
    train = 1
    test = 2


class Image:

    def __init__(self, path, folderName):
        self.path = path
        self.folderName = folderName
        self.data = []
        self.type = Type

    def set_data(self):
        self.data = self.folderName.find('1')


class Modes(enum.Enum):
    test = 1
    first_run = 2
    camera = 3
    train = 4
