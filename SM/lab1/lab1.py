import numpy as np
import matplotlib.pyplot as plt


def img_to_uint_8(img):
    img = img * 255
    img.astype('uint8')


def img_to_float(img):
    return img / 255.


def to_gray_scale_1(img):
    gray_image = 0.299 * img[:, :, 0]
    gray_image += 0.587 * img[:, :, 1]
    gray_image += 0.114 * img[:, :, 2]
    return gray_image


def to_gray_scale_2(img):
    gray_image = 0.216 * img[:, :, 0]
    gray_image += 0.7152 * img[:, :, 1]
    gray_image += 0.0722 * img[:, :, 2]
    return gray_image


def filter_one_color(img, color_idx):
    indices = [0, 1, 2]
    indices.remove(color_idx)
    for color_idx in indices:
        img[:, :, color_idx] = 0
    print()
    return img


def zad1():
    for img_name in ["B01.png", "B02.jpg"]:
        img = plt.imread(img_name)
        im_min = np.min(img)
        im_max = np.max(img)

        fig, axs = plt.subplots(3, 3)

        axs[0, 0].imshow(img)

        axs[0, 1].imshow(to_gray_scale_1(img), cmap=plt.cm.gray,
                         vmin=im_min, vmax=im_max)
        axs[0, 2].imshow(to_gray_scale_2(img), cmap=plt.cm.gray,
                         vmin=im_min, vmax=im_max)
        axs[1, 0].imshow(img[:, :, 0], cmap=plt.cm.gray,
                         vmin=im_min, vmax=im_max)
        axs[1, 1].imshow(img[:, :, 1], cmap=plt.cm.gray,
                         vmin=im_min, vmax=im_max)
        axs[1, 2].imshow(img[:, :, 2], cmap=plt.cm.gray,
                         vmin=im_min, vmax=im_max)

        axs[2, 0].imshow(filter_one_color(img.copy(), 0))
        axs[2, 1].imshow(filter_one_color(img.copy(), 1))
        axs[2, 2].imshow(filter_one_color(img.copy(), 2))

        plt.show()


def cut_image(img, w1, w2, k1, k2):
    return img[w1:w2, k1:k2].copy()


def zad2():
    img_name = "B02.jpg"
    img = plt.imread(img_name)
    img = cut_image(img, 400, 600, 500, 700)
    im_min = np.min(img)
    im_max = np.max(img)

    plt.imshow(img)
    plt.savefig("zad_2_1.jpg")

    plt.imshow(to_gray_scale_1(img), cmap=plt.cm.gray,
               vmin=im_min, vmax=im_max)
    plt.savefig("zad_2_2.jpg")

    plt.imshow(to_gray_scale_2(img), cmap=plt.cm.gray,
               vmin=im_min, vmax=im_max)
    plt.savefig("zad_2_3.jpg")

    plt.imshow(img[:, :, 0], cmap=plt.cm.gray,
               vmin=im_min, vmax=im_max)
    plt.savefig("zad_2_4.jpg")

    plt.imshow(img[:, :, 1], cmap=plt.cm.gray,
               vmin=im_min, vmax=im_max)
    plt.savefig("zad_2_5.jpg")

    plt.imshow(img[:, :, 2], cmap=plt.cm.gray,
               vmin=im_min, vmax=im_max)
    plt.savefig("zad_2_6.jpg")

    plt.imshow(filter_one_color(img.copy(), 0))
    plt.savefig("zad_2_7.jpg")

    plt.imshow(filter_one_color(img.copy(), 1))
    plt.savefig("zad_2_8.jpg")

    plt.imshow(filter_one_color(img.copy(), 2))
    plt.savefig("zad_2_9.jpg")


if __name__ == '__main__':
    zad1()
    zad2()
