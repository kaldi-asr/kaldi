# Idlak Speech Synthesis Server

This server runs an RESTful API based on Python Flask.

When setting up the server a single user with Admin permissions is created.



## API

Functions that get only be run by the admin users have been indicated as such.

Every request must have the authentication header, ```bearer``` for password, and ```awt``` for web authentication tokens.

### Authentication

**Retrieve an authentication token**
```
GET /awt
```

*Response*
```
{
    'awt': ...
}
```

**Expire the authentication token**
```
DELETE /awt
```

### Users

**Retrieve a list of currently registered users.**
```
(ADMIN) GET /users
```

*Response*
```
{
    'users' : [
        {
           'uid' : ... ,
           'admin' : (true|false)
        },
        ...
    ]
}
```

**Create a new user account (with or without admin privileges), or change an existing account's admin status.**
```
(ADMIN) POST /users
{
    (optional) 'admin' : (true|false),
    (optional) 'uid' : ...
}
```

*Response*
```
{
    'uid' : ... ,
    'password' : ...
}
```

**Reset a password**
```
(ADMIN) POST /users/<uid>/password
```

*Response*
```
{
    'password' : ...
}
```

**Delete a user**
```
(ADMIN) DELETE /users/<uid>
```

*Response*
```
{
    'success' : (true|false)
}
```

### Voices

**Get available voices**
```
GET /voices
```
Options will filter results
```
language : language (ISO 2 letter code)
accent : 2 letter accent code
gender : male|female
```

*Response*
```
{
    'Voices' : [
        <voice_id>,
        ...
    ]
}
```

**Get voice details**
```
GET /voices/<voice_id>
```

*Response*
```
{
    'language' : ... ,
    'accent' : ... ,
    'gender' : ... ,
}
```

### Speech Synthesis

**Synthesise speech**
```
POST /speech
{
    "voice" : <voice_id>,
    (optional default=false) "streaming" : (true|false),
    (optional default=wav) "audio format" : (wav|ogg|mp3),
    "text" : ...
}
```

*Response*
```
TBD
```