## Battle System

## Log System

In order to achieve our GDD in LogSystem section, i've created the concept of implementation LogSystem Component in Sumobot. There are properties (sessions and current_session), and methods or behaviour to handle logging in gameplay session and all events given from Players and their robots.

### LogManager component

A structured log is our main purpose since we want to use the data and feed to our ML. Players may have multiple session or battle when they are playing Sumobot, therefore we need simple database to capture players' log. 

### Sessions 
is a set of gameplayID and the LogSession, storing metadata, Players action, and Robot information

<img width="538" alt="Image" src="https://github.com/user-attachments/assets/e56d4457-5e74-4568-b58f-47ad8ab76430" />

### Current Session 

is a LogSession itself. When the Session is ended, it will be added to the Sessions

### LogManager methods / behaviour

- Gameplay State

     <img width="413" alt="Image" src="https://github.com/user-attachments/assets/80bb920d-bc45-410e-895c-2a13f88be29b" />

- Global State

     <img width="158" alt="Image" src="https://github.com/user-attachments/assets/9af6fcf8-bb51-4f9f-b146-929eca076d0b" />

### LogManager usecase

<img width="532" alt="Image" src="https://github.com/user-attachments/assets/d7f697fd-f727-44ab-906b-6cf4608f4c79" />


## Networking

To support multiplayer setup, we will use Mirror Networking (formerly UNET). Mirror is a system for building multiplayer capabilities for Unity games. It is built on top of the lower level transport real-time communication layer, and handles many of the common tasks that are required for multiplayer games. While the transport layer supports any kind of network topology, Mirror is a server authoritative system; although it allows one of the participants to be a client and the server at the same time, so no dedicated server process is required. Working in conjunction with the internet services, this allows multiplayer games to be played over the internet with little work from developers.

Mirror is focused on ease of use and iterative development and provides useful functionality for multiplayer games, such as:

- Message handlers
- General purpose high performance serialization
- Distributed object management
- State synchronization
- Network classes: Server, Client, Connection, etc.

Source: https://mirror-networking.gitbook.io/docs/manual/general

### Networking Role

A role represents what device that supposed to give an act.

<img width="396" alt="Image" src="https://github.com/user-attachments/assets/5907691e-8614-45f5-8f5f-bdc864090467" />

- Server: Act as a servant that provides interaction between clients & host
- Host: In this case, this role act as a controller of the clients
- Client: Mostly known as the player common itself. In Private Local Server, a Host can be client as well.

#### Private Local Server - Local couch multiplayer

This method will use Local Network from one of clients' device. For example, in order to play game in multiplayer mode, clientA's device needs to enable hotspot wifi to be the Server then the other clients should be connected to the clientA. It allows other clients to access clientA device address. Host role will be assigned to one of clients that initiated to create a room. 

**For now, if we want to enable connection using a wifi router**, we need to configure port-forwardding in router setting and point the Mirror's port.
This will allow incoming connection from clients that want to access to the private server address

#### Dedicated Server - Online multiplayer (Future enhancement)

Contradicts with the Private Local Server, the server itself will be built as an engine that should deployed on a machine. The machine should has static public IP to allow players around the world access the multiplayer mode

### Scene management

<img width="1066" alt="Image" src="https://github.com/user-attachments/assets/7302332d-94a4-4304-b7be-ddd10a6cbf01" />


**Offline scene** is a scene that players didn't involved with connectivity, such as Splash, Main Menu, select game modes, turn mode, control mode.

**Online scene** is a scene where players has joined to the server, either being host / client.


### Use Case

<img width="308" alt="Image" src="https://github.com/user-attachments/assets/300bd0c7-bffe-4f54-a7c6-ea40bb084a3a" />

#### Explanations:

1. After player pressed **Play** button, the player would enter Online scene and client will be the default role.
2. Player entered Online scene, and Search for a room. If any, it will be joined to the existing room. If not, this Player would be switched to be a Host and create a Room.
3. There will be an event listener to the Room whether someone has joined or not, then it will let us know which the room is Full (2 players). To decide whether all players are ready or not, we will use the Room metadata. A countdown might be added if theres no other clients joining the room.
4. After all players are ready there will be a countdown to help server syncing the players state.
5. Then a gameplay scene is running, pre-battle state must be entered by all players
6. After the battle is ended, there will be post-battle showcase to sums-up the battle. There might be a Rematch button.