from twilio.rest import Client

def message():
# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
    account_sid = 'AC017af98911276918488cee4b304b6a21'
    auth_token = '4f04cdc47763a86e902f1f4f89b9b61c'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                     body=" **** YOU LEFT A LIGHT ON **** ",
                     from_='+12056273288',
                     to='+917791010556'
                                    )

    print(message.sid)
