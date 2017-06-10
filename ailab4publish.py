import csv
from datetime import date

import cognitive_face as FaceAPI
import vk
from progressbar import ProgressBar

apikey = 'FaceAPIKey'

DEFAULT_VK_PHOTO_200_ORIG_LINK = 'http://vk.com/images/camera_a.gif'
vkAccessToken = 'VKKey'
vkRootUserId = 3796212


def calc_age(birth_date):
    today = date.today()
    day_month_year = birth_date.split('.')
    born = date(int(day_month_year[2]), int(day_month_year[1]), int(day_month_year[0]))
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def getinfo(photo_link):
    return FaceAPI.face.detect(photo_link, False, False, 'age,gender,emotion')


def predict_age(user_photo_link):
    res = FaceAPI.face.detect(user_photo_link, False, False, 'age')
    if len(res) == 0:
        raise Exception
    return round(res[0]['faceAttributes']['age'])


def average_happiness(meta_info):
    if len(meta_info) == 0:
        raise Exception
    number_of_informative_elems = 0
    sum_of_info = 0
    for info in meta_info:
        if len(info) != 0:
            number_of_informative_elems += 1
            sum_of_info += info[0]['faceAttributes']['emotion']['happiness']
    if number_of_informative_elems == 0:
        raise Exception
    return sum_of_info / number_of_informative_elems


def collect_training_set():
    FaceAPI.Key.set(apikey)

    session = vk.Session(access_token=vkAccessToken)
    api = vk.API(session)
    friends = api.friends.get(user_id=vkRootUserId, fields='sex,bdate,photo_200_orig')

    training_csv = open('./training_set' + str(vkRootUserId) + '.csv', 'w', encoding='cp1251')
    training_csv_headers = ['Uid', 'Happiness', 'Age', 'Sex', 'numOfFriends']

    writer = csv.writer(training_csv, delimiter=';', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(training_csv_headers)

    i = 0
    p_bar = ProgressBar()
    for friend in p_bar(friends):

        try:
            friend_photos = api.photos.getAll(owner_id=int(friend['uid']))
            training_set_csv_row = []

            meta_info = []
            for friend_photo in friend_photos[1:]:
                if 'src_big' in friend_photo:
                    meta_info.append(getinfo(friend_photo['src_big']))

            # Row format: Uid | Happiness | Age | Sex | numberOfFriends

            training_set_csv_row.append(friend['uid'])
            training_set_csv_row.append(average_happiness(meta_info))

            if 'bdate' in friend:
                if len(friend['bdate'].split('.')) == 3:
                    training_set_csv_row.append(calc_age(friend['bdate']))
                else:
                    training_set_csv_row.append(predict_age(friend['photo_200_orig']))
            else:
                training_set_csv_row.append(predict_age(friend['photo_200_orig']))

            training_set_csv_row.append(friend['sex'])
            training_set_csv_row.append(len(api.friends.get(user_id=int(friend['uid']))))

            writer.writerow(training_set_csv_row)
            i += 1

        except Exception:
            continue

        if i == 100:
            break

    training_csv.close()
    print("training set contains ", i, " elements")


# start
collect_training_set()
