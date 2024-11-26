import numpy as np
import matplotlib.pyplot as plt

#source: https://www.mit.edu/course/16/16.070/www/project/PF_kalman_intro.pdf

class KalmanFilter:
    def __init__(self,A,B,H,R,Q,x0,P0):
        self.A = A #relates the state x at k-1 to the state x at k, assumed constant
        self.B = B #relates the optional control input u to state x
        self.H = H #relates the measurement z to the state x
        self.R = R #Measurement Covariance
        self.Q = Q #State uncertainty
        self.x = x0 #initial state
        self.P = P0 #initial error estimate

    def prediction(self, u):
        # predict state x and Covariance P
        self.x = np.dot(self.A,self.x) + np.dot(self.B,u)
        self.P = np.dot(np.dot(self.A,self.P),self.A.T) + self.Q
        return self.x
        

    def correction(self,z):
        # calculate the Kalman gain
        N = np.dot(np.dot(self.H,self.P),self.H.T) + self.R
        K = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(N))

        # update the state
        y = z - np.dot(self.H,self.x)
        self.x = self.x + np.dot(K,y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K,self.H),self.P)
        return self.x

# Example usage

# say you have a vehicle with sinusoidal acceleration, initial velocity zero, initial position 0. 
# Every 2 seconds, you get a GPS - measurement and an accelerometer reading

# generate sample measurements
def generateAccelerations():
    a = []
    for t in range(0,18,2):
        a += [np.sin(0.2*t)]

    v = [0]
    s = [np.array([0])]
    for i in range(1, len(a)):
        v += [a[i]*dt+v[i-1]]
        s += [v[i]*dt + s[i-1]]
    return a,s

def generateMeasurements(a,dt):
    v = [0]
    s = [np.array([0])]
    mean = 0
    std = 1
    for idx in range(1, len(a)):
        v += [a[idx]*dt+v[idx-1]]
        s += [np.random.normal(mean, std,1) + v[idx]*dt + s[idx-1]]
    
    y = []
    std = 0.4
    for x in a:
        y += [np.random.normal(mean, std, 1) + x]
    return y,s

# set Kalman parameters
dt = 2 # time interval 2s
A = np.array([[1, dt, 0], [1, dt, dt], [0, 0, 1]]) 
B = np.array([[0.1], [0.1], [0]])     
H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    
R = np.array([[2e-1],[0],[3e-1]])      
Q = np.array([[5e-1, 0,0], [0, 1e-1,0],[0,0,7e-1]]) 


# theoretical accelerations
accel,pos = generateAccelerations() 

# measured accelerations and positions (theoretical values with white noise)
meas_a, meas_pos = generateMeasurements(accel,dt)

t = [0,2,4,6,8,10,12,14,16]

plt.figure(1)
plt.plot(t,accel,label="theoretical acceleration")
plt.plot(t,meas_a,label="measured acceleration")
plt.legend(loc="upper left")
plt.title("theoretical vs measured accelerations")
plt.xlabel("time [s]")
plt.ylabel("accelerations [m/s^2]")
plt.show()

plt.figure(2)
plt.plot(t,pos,label="theoretical position")
plt.plot(t,meas_pos,label="measured position")
plt.legend(loc="upper left")
plt.title("theoretical vs measured positions")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.show()

# Initial state and covariance
x0 = np.array([[0], [0], [0]]) 
P0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 

# Create Kalman Filter instance
kf = KalmanFilter(A, B, H, R, Q, x0, P0)

# Predict and update with the control input and measurement
u = np.array([[0]])  

predicted_pos = []
predicted_vel = []

# Predict step
predicted_state = kf.prediction(u)

# Update step
updated_state = kf.correction(x0)

predicted_pos.append(updated_state[0])
predicted_vel.append(updated_state[1])

for i in range(1,len(t)):
    p = meas_pos[i]
    v = meas_a[i]*dt+predicted_vel[i-1]
    a = meas_a[i]
    z = np.array([[p[0]], [v[0]], [a[0]]])

    # Predict step
    predicted_state = kf.prediction(u)

    # Update step
    updated_state = kf.correction(z)

    predicted_pos.append(updated_state[0])
    predicted_vel.append(updated_state[1])

plt.figure(3)
plt.plot(t,predicted_pos,label="predicted position")
plt.plot(t,meas_pos,label = "measured position")
plt.legend(loc="upper left")
plt.title("predicted vs measured positions")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.show()




    
