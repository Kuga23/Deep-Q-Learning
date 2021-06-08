"""Pacman, classic arcade game"""

##############################
### Importation de modules ###
##############################

from turtle import * # cause des 16 'erreurs' non problématiques
from tkinter import *
from random import choice
from math import sqrt
from time import sleep, ctime
import os
import string
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


############################
### Paramètre par défaut ###
############################

CONTROL_DEFAULT = {"^": "Up", ">": "Right", "v":"Down", "<":"Left"}

PACMAN_DEFAULT =  ["Pacman", [10, 10], [ 0,  0], "yellow"]

GHOSTS_DEFAULT = [["Blinky", [ 1,  1], [ 1,  0],    "red"],
                  [ "Pinky", [ 1, 15], [ 0,  1],   "pink"],
                  [  "Inky", [17,  1], [ 0, -1],   "cyan"],
                  [ "Clyde", [17, 15], [-1,  0], "orange"]]

COLOR_DEFAULT = {"background": "black", "maze": "blue", "pacgomme": "white", "text": "white"}

MAP_DEFAULT = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


####################################
### Fonctions principales du jeu ###
####################################

class PacmanEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    ######################
    ### Initialisation ###
    ######################

    def __init__(self,
                SIZE = 40,  # taille d'une case
                STEP_DIV = 1, # nombre de pas par case
                FPS = 24, # nombre d'image par seconde
                TIMER_MAX = 600, # chronometre, 0 = infini
                MAP = MAP_DEFAULT, # carte du labyrinthe
                PACMAN_INIT = PACMAN_DEFAULT, # description des personnages, Pacman en premier
                GHOSTS_INIT = GHOSTS_DEFAULT, # description des personnages, Pacman en premier
                COLOR_PANEL = COLOR_DEFAULT, # couleur de l'environnement
                TEXT_VECT = [7, 5], # position du texte
                PLAYER_HUMAIN = False, # utilisation des controles manuels
                CONTROL = CONTROL_DEFAULT, # definition des touches
                SHOW = False, # rendu visuel via Turtle,
                SAVE = False): # enregistrement des rendus au format eps

        # Initialisation des paramètres de base
        super(PacmanEnv, self).__init__()
        self.FPS = FPS
        self.STEP_DIV = STEP_DIV
        self.PLAYER_HUMAIN = PLAYER_HUMAIN
        if PLAYER_HUMAIN and not SHOW:
            print("ERREUR: It can't work if you don't see")
        self.SHOW = SHOW
        self.SAVE = SAVE
        self.COLOR = COLOR_PANEL
        self.TIMER_MAX = TIMER_MAX

        self.STAT = {"score": 0, "time": TIMER_MAX*STEP_DIV}
        self.WINNER = "NONE"
        self.EAT = "NONE"

        # Initialisation de la carte
        self.MAP_origine = MAP
        self.MAP = MAP
        self.SIZE = SIZE
        if SIZE % STEP_DIV != 0:
            print("ERREUR: SIZE must be multiple of STEP_DIV")
        self.STEP = int(SIZE/STEP_DIV) # DEFAULT = 10
        self.MAX_SCORE = sum(sum(MAP,[])) # DEFAULT = 160
        self.DIM_X = len(MAP)    # DEFAULT = 20
        self.DIM_Y = len(MAP[0]) # DEFAULT = 20

        # Initialisation des personnages
        self.PACMAN_INIT = PACMAN_INIT
        self.GHOSTS_INIT = GHOSTS_INIT
        Character = self.make_class()
        self.PACMAN = Character(PACMAN_INIT)
        self.GHOSTS = [Character(GHOST_INIT) for GHOST_INIT in GHOSTS_INIT]

        # Initialisation des controles
        self.list_aim = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        if self.PLAYER_HUMAIN:
            listen()
            onkey(lambda: self.PACMAN.change(self.list_aim[0]), CONTROL[">"])
            onkey(lambda: self.PACMAN.change(self.list_aim[1]), CONTROL["<"])
            onkey(lambda: self.PACMAN.change(self.list_aim[2]), CONTROL["^"])
            onkey(lambda: self.PACMAN.change(self.list_aim[3]), CONTROL["v"])

        # Initialisation de l'affichage
        if self.SHOW:
            self.path = Turtle(visible=False)
            self.writer = Turtle(visible=False)
            setup((self.DIM_Y+1)*SIZE, (self.DIM_X+1)*SIZE, 370, 0)
            hideturtle()
            tracer(False)

            # Initialisation du text
            self.writer.goto(TEXT_VECT[0]*SIZE, TEXT_VECT[1]*SIZE)
            self.writer.color(self.COLOR["text"])
            self.writer.write("LOADING...")

            # Initialisation de l'enregistrement
            if SAVE :
                self.nbr = 0
                self.origine = os.getcwd()
                path = os.getcwd() +"\\"+ ctime().replace(" ", "_").replace(":", "_")
                try:
                    os.mkdir(path)
                    os.chdir(path)
                except OSError:
                    print ("Creation of the directory %s failed" % path)
                else:
                    print ("Successfully created the directory %s" % path)

        ######################
        ### Ajout pour GYM ###
        ######################

        # Initialisation
        self.current_step = 0
        self.score_prev = self.STAT["score"]

        self.reward_range = (0, self.MAX_SCORE) 
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-2, 2, (self.DIM_X, self.DIM_Y))

    def _reset(self):
        self.STAT = {"score": 0, "time": self.TIMER_MAX*self.STEP_DIV}
        self.WINNER = "NONE"
        self.EAT = "NONE"
        self.MAP = self.MAP_origine
        Character = self.make_class()
        self.PACMAN = Character(self.PACMAN_INIT)
        self.GHOSTS = [Character(GHOST_INIT) for GHOST_INIT in self.GHOSTS_INIT]
        self.score_prev = self.STAT["score"]
        self.current_step = 0
    
    def _next_observation(self):
        obs = np.array(self.MAP)
        obs = 3 - np.where(obs==0, 3, obs) 

        v = self.comb_vect(self.PACMAN.position, [1, 1], coef2=self.SIZE/2)
        [i, j] = self.vect_to_index(v)
        obs[i, j] = -1 

        for GHOST in self.GHOSTS:
            v = self.comb_vect(GHOST.position, [1, 1], coef2=self.SIZE/2)
            [i, j] = self.vect_to_index(v)
            obs[i, j] = -2
        return obs # -2 : fantome, -1 = pacman, 0 : mur, 1 : rien, 2 : pacgomme
    
    def reward_fct(self):
        reward = self.STAT["score"]# - self.score_prev
        self.score_prev = self.STAT["score"]
        return reward

    def _step(self, action):
        self._take_action(action)
        self.current_step +=1

        self.next_step()

        obs = self._next_observation()
        reward = self.reward_fct()
        done = self.WINNER !="NONE"

        return obs, reward, done, {}
        
    def _take_action(self, action):
        self.PACMAN.change(self.list_aim[action])

    def _render(self, mode='humain', close = False):
        map = np.array(self.MAP)
        rendu = np.where(map==0, "%", 0)
        rendu = np.where(map==1, ".", rendu)
        rendu = np.where(map==2, " ", rendu)
        [i, j] = self.vect_to_index(self.comb_vect(self.PACMAN.position, [1, 1], coef2=self.SIZE/2))
        rendu[i, j] = "O" 
        for GHOST in self.GHOSTS:
            [i, j] = self.vect_to_index(self.comb_vect(GHOST.position, [1, 1], coef2=self.SIZE/2))
            rendu[i, j] = "X"
        print(rendu)
        if self.WINNER!=0:
            self.text = "score : " + str(self.STAT["score"]) + "\ntime : " + str(int(self.STAT["time"]/self.STEP_DIV))
            if self.WINNER == self.PACMAN.name:
                self.text = self.text + "\n\nGAME OVER : \nYou won " + self.WINNER
            else:
                self.text = self.text + "\n\nGAME OVER : \n" + self.WINNER + " ate you" 
            print(self.text)
        

        


    ######################################
    ### Fonctions utiles pour vecteurs ###
    ######################################

    def comb_vect(self, vect1, vect2, coef1=1, coef2=1): # combinaison linéaire de deux vecteurs
        return ([ int(coef1*vect1[0] + coef2*vect2[0]), int(coef1*vect1[1] + coef2*vect2[1]) ])

    def distance(self, vect1, vect2): # donne la distance entre deux vecteurs
        d = self.comb_vect(vect2, vect1, coef2 = -1)
        return int(sqrt(d[0]*d[0] + d[1]*d[1]))

    def vect_to_index(self, vect): # donne l'index d'un vecteur
        x = float((vect[0] + self.DIM_X*self.SIZE/2) // self.SIZE) * self.SIZE - self.DIM_X*self.SIZE/2
        y = float((vect[1] + self.DIM_Y*self.SIZE/2) // self.SIZE) * self.SIZE - self.DIM_Y*self.SIZE/2

        x = self.DIM_X/2 + x/self.SIZE
        y = self.DIM_Y/2 - 1 - y/self.SIZE
        return [int(y), int(x)]

    def index_to_vect(self, index): # donne le vecteur associé à un index
        x = (self.DIM_X/2 - (index[0]+1)) * self.SIZE
        y = (index[1] - self.DIM_Y/2) * self.SIZE
        return [int(y), int(x)]

    def valid(self, vect): # verifie la validité d'une position avant déplacement
        for add in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            index = self.vect_to_index(self.comb_vect(vect, add, coef2 = self.SIZE-1))
            if self.MAP[index[0]][index[1]] == 0:
                return False
        return True

    def square(self, vect): # dessine un carrée en position du vecteur
        self.path.up()
        self.path.goto(vect[0],vect[1])
        self.path.down()
        self.path.begin_fill()
        self.path.forward(self.SIZE)
        self.path.left(90)
        self.path.forward(self.SIZE)
        self.path.left(90)
        self.path.forward(self.SIZE)
        self.path.left(90)
        self.path.forward(self.SIZE)
        self.path.left(90)
        self.path.end_fill()

    ##############################
    ### Classe des personnages ###
    ##############################

    def make_class(self): # crée la classe des personnages

        outer_self = self # permet d'appeler les methodes de l'environement

        class Character: 

            def __init__(self, attribute):
                self.name = attribute[0] 
                self.position = outer_self.index_to_vect(attribute[1])
                if not outer_self.valid(self.position):
                    print("ERREUR : initial position of "+self.name+" invalid")
                self.aim = attribute[2]
                self.new_aim = self.aim
                self.color = attribute[3]
            
            def change(self, new_aim): # change la direction
                self.new_aim = new_aim

            def move(self): # fait un pas dans la direction si possible
                new_position = outer_self.comb_vect(self.position, self.new_aim, coef2=outer_self.STEP)
                if outer_self.valid(new_position): # tente de change de direction
                    self.position = new_position
                    self.aim = self.new_aim # valide la nouvelle direction 
                else:
                    new_position = outer_self.comb_vect(self.position, self.aim, coef2=outer_self.STEP)
                    if outer_self.valid(new_position): # tente d'avancer sans changer de direction
                        self.position = new_position
                    else:
                        return True # Imobilité
                return False
            
            def draw(self): # dessine le personnage avec un cercle de sa couleur
                up()
                v = outer_self.comb_vect(self.position, [1, 1], coef2=outer_self.SIZE/2)
                goto(v[0],v[1])
                dot(outer_self.SIZE, self.color)
        return(Character)


    #####################
    ### Boucle de jeu ###
    #####################

    def play(self):
        self.affichage(BEGINING=True)
        while self.WINNER == "NONE":
            self.next_step()
            if self.SHOW:
                self.affichage()
                if self.SAVE :
                    getcanvas().postscript(file=str(self.nbr)+".eps")
                    self.nbr +=1
                sleep(1/self.FPS) # attend le prochain pas de temps
        self.exit()
        return

    def exit(self):
        if self.SAVE :
            os.chdir(self.origine)
        done()
        print('fini')
        return

    def next_step(self):
        # Evolution du Timer
        self.STAT["time"] += -1
        if self.STAT["time"] == 0:
            self.WINNER = "The timer" # Temps écoulé
            return
        
        # Evolution de Pacman
        self.PACMAN.move()

        [i, j] = self.vect_to_index(self.comb_vect(self.PACMAN.position, [1, 1], coef2=self.SIZE/2))
        if self.MAP[i][j] == 1:
            self.MAP[i][j] = 2 # Pacman mange une pacgomme
            self.EAT = [i,j]
            self.STAT["score"] += 1
        if self.STAT["score"] == self.MAX_SCORE:
            self.WINNER = self.PACMAN.name # victoire de Pacman

        # Evolution des fantomes
        for GHOST in self.GHOSTS :
            if GHOST.move(): # si le fantome n'arrive pas à avancer, il change aléatoirement de direction
                GHOST.change(choice(self.list_aim))

            if self.distance(GHOST.position, self.PACMAN.position) < self.SIZE:
                self.WINNER = GHOST.name # victoire des fantomes


    ############################
    ### Affichage via Turtle ###
    ############################
    
    def affichage(self, BEGINING=False):
        clear()
        
        if BEGINING:
            # dessine le font
            bgcolor(self.COLOR["background"])

            # dessine le labyrinthe
            self.path.color(self.COLOR["maze"])

            for i in range(self.DIM_X):
                for j in range(self.DIM_Y):
                    if self.MAP[i][j] > 0:
                        self.square(self.index_to_vect([i,j]))

                        # dessine les pacgommes
                        if self.MAP[i][j] == 1:
                            self.path.up()
                            v = self.comb_vect(self.index_to_vect([i,j]), [1, 1], coef2=self.SIZE/2)
                            self.path.goto(v[0],v[1])
                            self.path.dot(self.SIZE/10, self.COLOR["pacgomme"])
        
        # efface les pacgommes
        if self.EAT != "NONE":
            self.square(self.index_to_vect(self.EAT))
            self.EAT = "NONE"
    
        # dessine les personnages
        self.PACMAN.draw()
        for GHOST in self.GHOSTS :
            GHOST.draw()

        # dessine le texte
        self.writer.undo()
        self.text = "score : " + str(self.STAT["score"]) + "\ntime : " + str(int(self.STAT["time"]/self.STEP_DIV))
        if self.WINNER == self.PACMAN.name:
            self.text = self.text + "\n\nGAME OVER : \nYou won " + self.WINNER
        elif self.WINNER != "NONE":
            self.text = self.text + "\n\nGAME OVER : \n" + self.WINNER + " ate you" 
        self.writer.write(self.text)

        update()



#####################
### Zone de Tests ###
#####################

env= PacmanEnv()