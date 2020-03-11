import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import copy
from shapely.geometry import Polygon
import sys
import json
import time
import multiprocessing 
import warnings

gene = []
message = dict
pop = []
mutationRate = .2
inRouteMutationProb = .5
crossRouteMutationProb = .5

def miniGA(listt):
  
  cityLists = listt[1:len(listt)-1]
  
  iniCity = listt[0]
  
  
  
  mutationRate = .001
  warnings.simplefilter(action='ignore', category=FutureWarning)
  class City:
      def __init__(self, x, y):
          self.x = x
          self.y = y
      def distance(self, city):
        
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
      
      def __repr__(self):
          return "(" + str(self.x) + "," + str(self.y) + ")"



  class Fitness:
      def __init__(self, route):
          self.route = route
          self.distance = 0
          self.fitness= 0.0
      
      def routeDistance(self):
          if self.distance ==0:
              pathDistance = 0
              for i in range(0, len(self.route)):
                  fromCity = self.route[i]
                  toCity = None
                  if i + 1 < len(self.route):
                      toCity = self.route[i + 1]
                  else:
                      toCity = self.route[0]
                  pathDistance += distanceMatrix[fromCity][toCity]
                  self.distance = pathDistance
          return self.distance
      
      def routeFitness(self):
          if self.fitness == 0:
              self.fitness = 1 / float(self.routeDistance())
          return self.fitness

  def createRou(cityList):
      route = []
      routa = []
      route.append(0)
      routa = cityListi
      for i in routa:
          route.append(i)
      route.append(0)
      
      return route

  def createRoute(cityList):
      route = []
      routa = []
      route.append(0)
      routa = random.sample(cityListi, len(cityListi))
      for i in routa:
          route.append(i)
      route.append(0)
      
      return route

  def initialPopulation(popSize, cityList):
      population = []

      for i in range(0, popSize//2):
          population.append(createRou(cityList))
      for i in range(popSize//2,popSize):
          population.append(createRoute(cityList))    
      for i in population:
          if(i == None):
              del(i)
      return population


  def rankRoutes(population):
      
      fitnessResults = {}
      for i in range(0,len(population)):
          fitnessResults[i] = Fitness(population[i]).routeFitness()
      return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

  def selection(popRanked, eliteSize):
      selectionResults = []
      df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
      df['cum_sum'] = df.Fitness.cumsum()
      df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
      
      for i in range(0, eliteSize):
          selectionResults.append(popRanked[i][0])
      for i in range(0, len(popRanked) - eliteSize):
          pick = 100*random.random()
          for i in range(0, len(popRanked)):
              if pick <= df.iat[i,3]:
                  selectionResults.append(popRanked[i][0])
                  break
      return selectionResults

  def matingPool(population, selectionResults):
      matingpool = []
      for i in range(0, len(selectionResults)):
          index = selectionResults[i]
          matingpool.append(population[index])
      return matingpool



  def deconstruct(parent1):
      parent1 = parent1[1:len(parent1)-1]
      return parent1


  def reconstruct(new):
      par=[]
      par.append(0)
      for i in new:
          par.append(i)
      par.append(0)
      return par



  def breed(parent1, parent2):
      
      
      parent1 = deconstruct(parent1)
      parent2 = deconstruct(parent2)
      
      child = []
      childP1 = []
      childP2 = []
      
      geneA = int(random.random() * len(parent1))
      geneB = int(random.random() * len(parent1))
      
      startGene = min(geneA, geneB)
      endGene = max(geneA, geneB)

      for i in range(startGene, endGene):
          childP1.append(parent1[i])
          
      childP2 = [item for item in parent2 if item not in childP1]

      child = childP1 + childP2

      child = reconstruct(child)
      
      return child

  def breedPopulation(matingpool, eliteSize):
      children = []
      length = len(matingpool) - eliteSize
      pool = random.sample(matingpool, len(matingpool))

      for i in range(0,eliteSize):
          children.append(matingpool[i])
      
      for i in range(0, length):
          child = breed(pool[i], pool[len(matingpool)-i-1])
          children.append(child)
      return children
  
  def mutate(individual):
      individual = deconstruct(individual)
      for swapped in range(len(individual)):
          if(random.random() < mutationRate):
              
              swapWith = int(random.random() * len(individual))
              
              city1 = individual[swapped]
              city2 = individual[swapWith]
              
              individual[swapped] = city2
              individual[swapWith] = city1
      individual = reconstruct(individual)
      return individual

  def mutatePopulation(population, mutationRate):
      
      
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind])
        mutatedPop.append(mutatedInd)
    return mutatedPop
  def fixni(individual):
      
      copyIndi = copy.deepcopy(individual)
      unfinished= True

      while(unfinished):

        switched = False
        for swapped in range(1,len(individual)-1):
          
            for swapWith in range(1,len(individual)-1):
              individual = copy.deepcopy(copyIndi)
              
              city1 = individual[swapped]
              city2 = individual[swapWith]
              
              individual[swapped] = city2
              individual[swapWith] = city1

              
              if(Fitness(copyIndi).routeFitness()<Fitness(individual).routeFitness()):
                
                copyIndi = copy.deepcopy(individual)
                switched = True
        if(not switched):
          unfinished = False

      
      return copyIndi
  def fixrev(individual):
      
      copyIndi = copy.deepcopy(individual)
      unfinished= True

      while(unfinished):

        switched = False
        for swapped in range(1,len(individual)-1):
          
            for swapWith in range(swapped+1,len(individual)-2):

              individual = copy.deepcopy(copyIndi)
              indi = []
              
              part1 = individual[:swapped]
              if(swapped != swapWith):
                part2 = individual[swapped:swapWith+1]
                
                part2 = part2[::-1]
              else:
                part2 = []
              part3 = individual[swapWith+1:]
              indi += part1 
              indi += part2 
              indi += part3
              
              

              
              if(Fitness(copyIndi).routeFitness()<Fitness(indi).routeFitness()):
                
                copyIndi = copy.deepcopy(indi)
                switched = True
        if(not switched):
          unfinished = False

      
      return copyIndi

  def nextGeneration(currentGen, eliteSize, mutationRate):
      
      popRanked = rankRoutes(currentGen)
      selectionResults = selection(popRanked, eliteSize)
      matingpool = matingPool(currentGen, selectionResults)
      children = breedPopulation(matingpool, eliteSize)
      nextGeneration = mutatePopulation(children, mutationRate)
      return nextGeneration


  def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
      pop = initialPopulation(popSize, population)


      realCityList.append(initialCity)
      for i in range(0,len(cityList)):
          realCityList.append(cityList[i])
      for x in range(0,len(cityListi)+1):
          newmat = []
          for y in range(0,len(cityListi)+1):
              newmat.append(realCityList[x].distance(realCityList[y]))
              
          distanceMatrix.append(newmat)
         


      plt.scatter([l.x for l in cityList], [l.y for l in cityList])
      
      plt.scatter(initialCity.x, initialCity.y)
      plt.show(block=False)
      
      print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))


      progress = []
      progress.append(1 / rankRoutes(pop)[0][1])
      
      for i in range(0, generations):
          
          pop = nextGeneration(pop, eliteSize, mutationRate)
          progress.append(1 / rankRoutes(pop)[0][1])
      bestRouteIndex = rankRoutes(pop)[0][0]
      bestRoute = pop[bestRouteIndex]
      
      bestRoute1 = fixni(bestRoute)
      bestRoute3 = fixrev(bestRoute1)
      


      #print(bestRoute)
      #print(bestRoutee)
      bestRoutenum = []
      for l in bestRoute3:
          bestRoutenum.append(realCityList[l])
      print("Final distance: " + str(Fitness(bestRoute3).routeDistance()))
      
      plt.plot([l.x for l in bestRoutenum], [l.y for l in bestRoutenum], '.r-')
      plt.show()
      plt.plot(progress)
      plt.ylabel('Distance')
      plt.xlabel('Generation')
      plt.show()
      
      return bestRoute3
  
  initialCity = iniCity
  cityList = []
  cityListi = []
  realCityList= []
  newmat = []
  distanceMatrix = []

  cityList =  cityLists
  for i in range(0,len(cityList)):
      cityListi.append(i+1)


  #for i in range(0,25):
   #   cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))



  newroute3 = geneticAlgorithmPlot(population=cityList, popSize=200, eliteSize=20, mutationRate=1, generations=1)
  returnRoute = []
  for x in newroute3:

    returnRoute.append(listt[x])
  
  return returnRoute


def flip(p):
    return (random.random() < p)

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
        self.realDistance = 0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            notMin = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                thisDistance = DistanceMatrix[fromCity][toCity]
                pathDistance += thisDistance
                newarr = [x for i,x in enumerate(DistanceMatrix[toCity]) if i!=toCity]
                if(thisDistance != min(newarr)):
                  notMin +=1
                self.distance = pathDistance
        pathDistance += (notMin * 20)
        return self.distance 
    def realRouteDistance(self):
        if self.distance ==0:
            pathDistance = 0
            
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                
                pathDistance += DistanceMatrix[fromCity][toCity]
                
                self.realDistance = pathDistance
        
        
        return self.realDistance 
    def numberOfTrucks(self):
        trucks = 0
        for i in range(0,len(self.route)):
            if((self.route[i] == initialCity and (i == len(self.route) or i == 0))
             or  ((self.route[i] == initialCity )and  self.route[(i)] == initialCity)):
                trucks += 1

        return trucks
    def polyfitness(self):
      if(not tooSmall(self.route)):
        fitness = 0
        fitness = (30 * intersections(self.route) * lconstant)
      else:
        fitness = 5000 * lconstant
      return fitness
    def sizefitness(self):
      fitness = 0
      count, totDis = checkSize(self.route)
      fitness += (10000 *count * lconstant)
      fitness += (25 * totDis * lconstant)
      return fitness
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / (float(self.routeDistance()) + float(self.sizefitness())  )
            self.fitness 
        return self.fitness


def untangle(chrom):
  
  otherChrom = []
  otherChrom.append(chrom[0])
  otherChrom.append(chrom[2])
  otherChrom.append(chrom[1])
  otherChrom.append(chrom[3])

  if(Fitness(chrom).routeDistance() > Fitness(otherChrom).routeDistance()):
    
    return otherChrom
    

  return chrom

def untangleGene(gene):
  
  for x in range(0, len(gene)-4):
    temp = []
    temp += copy.deepcopy(gene[:x])
    for f in untangle(gene[x:(x+4)]):
      temp.append(f)
    temp += copy.deepcopy(gene[(x+4):])
    gene = copy.deepcopy(temp)
  
  return gene

def intersections(gene):
  
  temp = deconstruct(gene)
  
  listOfRoutes = []
  listOfPolys = []
  numOfInt = 0
  for x in temp:
    bestRoutenum = []
    for l in x:
        bestRoutenum.append(realCityList[l])
    listOfRoutes.append(copy.deepcopy(bestRoutenum))
  for y in listOfRoutes:
    if(len(y)>2):
      t = []
      for e in y:
        x = e.x
        y = e.y
        
        t.append([x,y])
      p = Polygon(t)
      p = p.buffer(0)
      listOfPolys.append(copy.deepcopy(p))
  if(len(listOfPolys) > 1):
    for t in range(0,len(listOfPolys)-1):
      for x in range(0,len(listOfPolys)-1):
      
        if((x != t) and listOfPolys[t].intersects(listOfPolys[x]) ):
          numOfInt += 1
        
  return numOfInt

def checkSize(gene) :
  
  temp = deconstruct(gene)
  

  totDis = 0
  count = 0
  countt = 0
  for x in temp:
    truth, ndis = tooBig(x, countt)
    if(truth):
      count += 1
      totDis += ndis
    countt += 1
  return count, totDis

def tooSmall(chromosome):
  truth = False
  
  if(len(chromosome) < 3):
    
    truth = True
  return truth
def tooBig(chromosome, truckNum):
  
  truth = False
  dis = 0
  for x in chromosome:
    dis += cargoList[x]
  
  if(dis > trucks[truckNum][0]):
    truth = True
    dis = dis - trucks[truckNum][0]
  
  return truth, dis

def createRoute(cityList):
    route = []
    routa = []
    
    routa = random.sample(cityListi, len(cityListi))
    for i in routa:
        route.append(i)
    
    
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    for i in population:
        if(i == None):
            del(i)
    return population

def getFItness(gene):

  ge = Fitness(gene).routeFitness()

  return ge
def rankRoutes(population):
    fitnessResults = {}
    fitnessiResults = {}

    pi = multiprocessing.Pool()
    fitnessResults = pi.map(getFItness,population)
    pi.close()
    
    
    fitnesResults = { i : fitnessResults[i] for i in range(0, len(fitnessResults) ) }
    


    return sorted(fitnesResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool



def reconstruct(new):
    
    par=[]
    
    for i in new:
        par.append(0)
        for x in i:
          par.append(x)
        par.append(0)
    
    return par

def addz(new):
  indexi = []
  for y in new:
    p = []
    for f in y:
      p.append(indexlist[f])
    indexi.append(p)
  
  s = []
  for x in new:
    p =[]
    p.append(0)
    
    p+=x
    p.append(0)
    s.append(p)

  return s, indexi

def deconstruct(parent1):
    
    temp = []
    count = 0
    parent1 = parent1[1:len(parent1)-1]
    for i in range(0,len(parent1)):
      
      if(parent1[i]==0 and parent1[i+1]):
        temp.append(parent1[count:(i-1)])
        count += (i -count)+1
    temp.append(parent1[count:])
    parent1 = temp
    
    return parent1

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)

    return children


def chromosomeMutation(a):
  
  x = copy.deepcopy(a)
  
  if(flip(mutationRate)):
    
    for y in x:
      
      y = inRouteMutation(y)
  
  if(flip(crossRouteMutationProb)):
    
    x = cross(x)
  
  if(flip(crossRouteMutationProb)): 
    
    x = join(x)
  
  if(flip(crossRouteMutationProb)):
    x = split(x)
  
  
  return x


def inRouteMutation(chromosome):

  chromosome = inv(chromosome)
  
  chromosome = trans(chromosome)
  
  #chromosome = ins(chromosome)
  return chromosome

def join(s):
  if(len(s) == minTrucks):
    return s
  if(len(s) == 1):
    return s
  chr1 = random.randint(0, len(s)-1)
  chr2 = chr1
  while(chr2 == chr1):
    chr2 = random.randint(0, len(s)-1)

  s[chr1] += s[chr2]
  s.remove(s[chr2])
  
  return s  
def split(f):
  if(len(f) == maxtrucks):
    return f

  chr1 = random.randint(0, len(f)-1)
  while(len(f[chr1]) <= 1):
    chr1 = random.randint(0, len(f)-1)

  cut1 = random.randint(1, len(f[chr1])-1)
  f.append(f[chr1][cut1:])
  f[chr1] = f[chr1][:cut1]

  
  return f  
def cross(genn):
  
  if(len(genn) <= 1):
    return(genn)
  chr1 = random.randint(0, len(genn)-1)
  chr2 = random.randint(0, len(genn)-1)
  while(chr1 == chr2):
    chr2 = random.randint(0, len(genn)-1)
  cut1a = random.randint(0, len(genn[chr1])-1)
  cut1b = random.randint(0, len(genn[chr1])-1)
  cut2a = random.randint(0, len(genn[chr2])-1)
  cut2b = random.randint(0, len(genn[chr2])-1)
  ct1a = min(cut1a,cut1b)
  ct1b = max(cut1a,cut1b)
  ct2a = min (cut2a,cut2b)
  ct2b = max(cut2a,cut2b)
  
  temp1a = genn[chr1][: ct1a]
  temp1b = genn[chr1][ct1b :]
  temp2a = genn[chr2][: ct2a]
  temp2b = genn[chr2][ct2b :]
  ins1 = genn[chr1][ct1a:ct1b]
  ins2 = genn[chr2][ct2a:ct2b]
  
  
  newa = temp1a
  newb = temp2a
  for i in ins2:
    newa.append(i)
  for i in ins1:
    newb.append(i)
  
  newa += temp1b
  newb += temp2b

  genn[chr1] = newa
  genn[chr2] = newb
  
  return genn


def trans(chromosome):
  if(len(chromosome) <=1):
    return chromosome
  cut = random.randint(0, len(chromosome)-1)
  cut3 = random.randint(0, len(chromosome)-1)
  cut1 = min(cut,cut3)
  cut2 = max(cut,cut3)

  temp = chromosome[cut1]
  chromosome[cut1] = chromosome[cut2]
  chromosome[cut2] = temp
  
  return chromosome


def ins(chromosome):
  
  
  cut = random.randint(0, len(chromosome)-1)
  cut2 = random.randint(0, len(chromosome)-1)
 
  
  chromosome.insert(cut2, chromosome[cut])
  chromosome.remove(chromosome[cut])
  
  
  return chromosome


def inv(chromosome):
  if(len(chromosome) <=1):
    return chromosome
  cut = random.randint(0, len(chromosome)-1)
  cut3 = random.randint(0, len(chromosome)-1)
  cut1 = min(cut,cut3)
  cut2 = max(cut,cut3)
  temp = []
  
  for i in range (cut1,cut2):
    temp.append(chromosome[i])
  for i in range(cut1,cut2):
    chromosome[i] = temp[len(temp)-(i-cut1)-1]
  
  
  return chromosome
  
def mutate(pop,mutationRate):

  pi = multiprocessing.Pool()
  popu = pi.map(chromosomeMutation,pop)
  pi.close()
  return popu

  




def unite(chromosome):
  
  order =[]
  chrom = []
  for c in chromosome:
    for x in c:
      chrom.append(x)
    order.append(len(c))
  chromosome = chrom
  
  return order,chromosome
def getIndexi(individual):
  newindividual = []

  for x in individual:
    newindividual.append(x[1:len(x)-1])
  print(newindividual)

  indexi = []
  for y in newindividual:
    p = []
    for f in y:
      p.append(indexlist[f])
    indexi.append(p)
  return indexi


def separate(chromosome, order):
  
  temp = []
  t = []
  count = 0
  for orders in order:
    
    temp.append(chromosome[count:(orders+count)])
    count += orders
  chromosome = temp
  
  return chromosome
def normalize(pop):
  
  popu = []
  for i in pop:
    temp = []
    num = int(random.random() * (maxtrucks - minTrucks)) + 3
    skip = len(i)//num
    start = 0
    curr = skip
    for c in range(1,num):
      temp.append(i[start:curr])
      start = curr
      curr += skip
    temp.append(i[start:])
    popu.append(temp)
  pops = reconPop(popu)
  
  return pops

def breed(genn, gen2):
  
  genn2 = copy.deepcopy(gen2)
  if(len(genn) == 1):
    return(genn)
  chr1 = random.randint(0, len(genn)-1)

  chr2 = random.randint(0, len(genn2)-1)
  
  cut1a = random.randint(0, len(genn[chr1])-1)
  cut1b = random.randint(0, len(genn[chr1])-1)
  
  ct1a = min(cut1a,cut1b)
  ct1b = max(cut1a,cut1b)
  
  
  temp1a = genn[chr1][: ct1a]
  temp1b = genn[chr1][ct1b :]
  
  ins1 = copy.deepcopy(genn[chr1][ct1a:ct1b])
  ins1c = copy.deepcopy(ins1) 
  
  for x in genn2:
    for y in x:
      if(y in ins1):
        
        ins1.remove(y)
        x.remove(y)
        
  
  cut2a =0
  if(len(genn2[chr2]) > 0):
    cut2a = random.randint(0, len(genn2[chr2])-1)
  
   
  temp2a = genn2[chr2][: cut2a]
  temp2b = genn2[chr2][cut2a :]
  


  newb = temp2a
  
  newb += ins1c
  
  newb += temp2b

  
  genn2[chr2] = newb

  for x in genn2:
    if(len(x) == 0):
      genn2.remove(x)
  


  return genn

def deconPop (pop):
  
  pi = multiprocessing.Pool()
  popu = pi.map(deconstruct,pop)
  pi.close()
  return popu

def reconPop (pop):
  
  pi = multiprocessing.Pool()
  popu = pi.map(reconstruct,pop)
  pi.close()
  return popu

def untangleGeneration(pop):
  p = multiprocessing.Pool()
  pop = p.map(untangleGene, pop)
  p.close()
  return pop

def nextGeneration(currentGen, eliteSize, mutationRate,generati):
    
    if(generati%20 == 0 ):
      currentGen = untangleGeneration(currentGen)
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    matingpool = deconPop(matingpool)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutate(children, mutationRate)
    nextGeneration = reconPop(nextGeneration)
    


    return nextGeneration


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    

    pop = normalize(pop)
    
    realCityList.append(initialCity)
    for i in range(0,len(cityList)):
        realCityList.append(cityList[i])
    for x in range(0,len(cityListi)+1):
        newmat = []
        for y in range(0,len(cityListi)+1):
            newmat.append(realCityList[x].distance(realCityList[y]))
            
        DistanceMatrix.append(newmat)
      
    alltime = pop[0]
    alltimefitness = Fitness(alltime).routeFitness()
    bestGen = 0
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        print("{\"percent\" : " + str(float(i)/generations) + "}" )
        
       
        pop = nextGeneration(pop, eliteSize, mutationRate,i)
        progress.append(1 / rankRoutes(pop)[0][1])

        bestRouteIndex = 0
        bestRoute = pop[bestRouteIndex]
        
        bestRouteFitness = Fitness(bestRoute).routeFitness()
        alltimefitness = Fitness(alltime).routeFitness()
        if(alltimefitness < bestRouteFitness):
            alltime = bestRoute
            bestGen = i
        


    
    bestRoute = alltime
    newBestRoute = untangleGene(bestRoute)
    bestRoutess = deconstruct(newBestRoute)
    for x in bestRoutess:
      dis = 0
      for i in x:
        dis += cargoList[i]
      
    bestRoutes, indexis = addz(bestRoutess)
    bestRoutenum = []
    for l in newBestRoute:
        bestRoutenum.append(realCityList[l])
    
    bestRoutenums = []
    for x in bestRoutes:
      bestRoutenum = []
      for l in x:
          bestRoutenum.append(realCityList[l])
      bestRoutenums.append(bestRoutenum)
    
    bestRoutenums2 = []
    for x in bestRoutenums:
        bestRoutenums2.append(miniGA(x))
    
    newrouteFinal =[]
    for y in bestRoutenums2:
      added = []
      for x in y:
        added.append(realCityList.index(x))
      newrouteFinal.append(added)

    indexis = getIndexi(newrouteFinal)



    cols = ['.b-','.g-','.r-','.c-','.m-','.y-','.k-']
    c =0
    for z in bestRoutenums2:
      col = random.randint(0, len(cols)-1)
      plt.plot([l.x for l in z], [l.y for l in z], cols[c])
      if(c<len(cols)):
        c +=1
      else:
        c=0

    plt.show()
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    ruts = []


    
    
    
    
    finalize(indexis)
    return 



cityList = []
cityListi = []
realCityList= []
newmat = []
indexis=[]
lconstant = 0
DistanceMatrix = []

trucks = []

pathin = str
pathout = str
ids =[]

if __name__ == "__main__":
    pathin = str(sys.argv[1])
    pathout = str(sys.argv[2])
    


def initialize():
  with open(pathin) as f:
     datas = f.read()
  
  y = json.loads(datas)
  message = copy.deepcopy(y)
  trucs =[]
  pen = y["trucks"]
  for i in pen:
    pair = i["capacity"], i["id"]
    trucs.append(pair)
  pair = [0, int(y["distribution_center"]["latitude"]*1000000), int(y["distribution_center"]["logitude"]*1000000)]
  ids.append(copy.deepcopy(pair))
  for x in y["stores"]:
    a  =x["latitude"]*1000000
    b = x["logitude"]*1000000
    c = x["id"]
    d=(1- x["value"])
    e= (d * x["capacity"])

    pair = [int(c), int(a), int(b), (e)]
    ids.append(copy.deepcopy(pair))

  izq =ids[0][1]
  bottom = ids[0][2]
  for p in ids:
    izq = min(izq, p[1])
    bottom = min(bottom, p[2])
  new =[]
  
  for x in ids:

    uno = x[1]-izq
    dos = x[2]-bottom
    if(x[0] == 0):
      y = [x[0],uno,dos,0]
    else:
      y = [x[0],uno,dos,x[3]]

    new.append(y)
  
  cityList = []
  for x in new:
    par = x[1],x[2],x[0],x[3]
    cityList.append(par)
  
  return cityList, message, trucs










def finalize(indexi):
  jOutputBaby = []
  copyY = copy.deepcopy(message)
  jOutputBaby = copy.deepcopy(copyY)
  jOutputBaby.pop("stores")
  count = 1
  rouss = []
  for x in indexi:
    
    
    route = []
    for y in x:
      for f in (copyY["stores"]):
        
        if(f["id"] == y ):
          
          route.append(f)
    for x in route:
      x.pop("timestamp")
      x.pop("capacity")
    dictii = {"id" : count, "name" : jOutputBaby["trucks"][count-1]["name"], "route" : route}
    count += 1
    rouss.append(dictii)
    
    
  jOutputBaby.update({"routes" : rouss})
  
  
  json.dump(jOutputBaby, open(pathout, 'w'), indent = 1)
  
  return




data, message, trucks = initialize()
initialCity = (City(x = data[0][0], y=data[0][1]))
cityLists = data[1:]
indexlist = []
for x in data:
  indexlist.append(x[2])
cargoList = []

for i in data:
  cargoList.append(i[3])

lconstant = len(cargoList)/30
maxtrucks = len(trucks)


for i in cityLists:
    cityList.append(City(i[0], i[1]))
for i in range(0,len(cityList)):
    cityListi.append(i+1)
population=cityList
minTrucks = 4
maxSizeRoute  = (len(cityLists)//maxtrucks) + 5
mutationRate = .2




geneticAlgorithmPlot(population=cityList, popSize=250, eliteSize=40, mutationRate= mutationRate, generations=15)






