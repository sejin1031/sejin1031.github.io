"""
* project_name : Rock Cissor Paper  
* Author       : Sejin_Eom           
* Version      : 1.0                
"""
''' # Bold random모듈을 import'''

import random
def r_c_p():
	print("가위바위보를 시작하겠습니다.")
	
	#setting start point of user and com win counts
	
	userwin = 0
	comwin = 0
	
	#loop for 10 games of Rock Cissor Paper
	
	for _ in range(10):
		print("무엇을 내시겠습니까?(가위1,바위2,보3) : ")
		user = int(input())
		com = random.randint(1,3)
		
		#Seperate with user's choice(1or2or3)

		if user == 1:

			#Seperate with com's choice(1or2or3)
			
			if com == 1:
				print("너는 가위 컴퓨터도 가위 무승부")
			elif com == 2:
				print("너는 가위 컴퓨터는 바위 컴승")
				comwin += 1
			elif com==3:
				print("너는 가위 컴퓨터는 보 너 승")
				userwin  += 1

		elif user == 2:

			#Seperate with com's choice(1or2or3)

			if com == 1:
				print("너는 바위 컴은 가위 너 승")
				userwin  += 1
			elif com==2:
				print("너는 바위 컴은 바위 무승부")
			elif com==3:
				print("너는 바위 컴은 보 컴승")
				comwin += 1
		elif user == 3:

			#Seperate with com's choice(1or2or3)

			if com == 1:
				print("너는 보 컴은 가위 컴승")
				comwin += 1
			elif com == 2:
				print("너는 보 컴은 바위 너 승")
				userwin  += 1
			elif com==3:
				print("너는 보 컴은 보 무승부")

	#Print the com and users win count!

	print("너가",userwin," 만큼이겼어요")
	print("컴이",comwin," 만큼이겻어요")

r_c_p()

