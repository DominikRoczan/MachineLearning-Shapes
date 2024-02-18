import datetime



class Cat:
    breed_list = ['dachowiec', 'maine coon', 'syberyjski']
    gender_dict = {0: 'Male', 1: 'Female'}
    color_dict = {
        'rudy': 'pregowany',
        'bialy': 'gładki',
        'czarny': 'w kropki'
    }
    full_color_line = list(zip(color_dict.keys(), color_dict.values()))

    def __init__(self):
        self.name = "------"
        self.gender = '------'
        self.birthday = 0
        self.breed = '-----'
        self.color = '-----'

    def years(self):
        return datetime.datetime.now().year - self.birthday

    def test(self):
        if self.years() > 15:
            return 'Heart scan'
        else:
            '-----'

    def description(self):
        return (
                '*' * 75 +
                '\n\n'
                f'Name: {self.name} '
                '\n'
                f'Gender: {self.gender}; Birthday:  {self.birthday}; Years: {self.years()}; Breed: {self.breed}; Color: {self.color}; '
                f'Test: {self.test()}'
                '\n'

        )


cat1 = Cat()
cat1.birthday = 2001
cat1.name = 'Mruczuś'
cat1.gender = Cat.gender_dict[0]
cat1.breed = Cat.breed_list[1]
cat1.color = Cat.full_color_line[0]

cat2 = Cat()
cat2.birthday = 2013
cat2.name = 'Pchełka'
cat2.gender = Cat.gender_dict[1]
cat2.breed = Cat.breed_list[2]
cat2.color = Cat.full_color_line[1]

print(cat1.description(), cat2.description())
