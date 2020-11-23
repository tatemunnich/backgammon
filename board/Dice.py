import random


class Dice:

	def __init__(self):
		self.die1 = random.randint(1, 6)
		self.die2 = random.randint(1, 6)

	def roll(self):
		self.die1 = random.randint(1, 6)
		self.die2 = random.randint(1, 6)

	def rollNoDoubles(self):
		self.roll()
		while self.isDoubles():
			self.roll()

	def setRoll(self, dice: tuple):
		die1, die2 = dice
		self.die1 = die1
		self.die2 = die2

	def getDiceDistances(self):
		if self.isDoubles():
			return 4*[self.die1]
		else:
			return [self.die1, self.die2]

	def getDice(self):
		return self.die1, self.die2

	def getDie1(self):
		return self.die1

	def getDie2(self):
		return self.die2

	def isDoubles(self):
		return self.die1 == self.die2

	def __str__(self):
		return "Dice: " + str(self.getDice())
