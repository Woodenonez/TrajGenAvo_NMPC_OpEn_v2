class PositionsFlow():
    def __init__(self, start, coupling_end, formation_end, decoupling_end):
        self.start = start
        self.coupling_end = coupling_end
        self.formation_end = formation_end
        self.decoupling_end = decoupling_end
        self.positions = [coupling_end, formation_end, decoupling_end]
        self.next = lambda : self.positions.pop(0)