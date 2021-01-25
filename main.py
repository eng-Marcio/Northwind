from DBReader import DBReader
from simple_drawer import SimpleDrawer

class Main:
    def __init__(self) :
        self.dataset = DBReader()
        self.simpleDrawer = SimpleDrawer(self.dataset)



if __name__ == "__main__":  # only execute if not imported
    main = Main()

