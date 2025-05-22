import torch
import textwrap
import copy
from PIL import Image, ImageFont, ImageDraw
from typing import cast
from .imagefunc import AnyType, log, get_resource_dir, tensor2pil, pil2tensor, image2mask


any = AnyType("*")

class SimpleTextImage:

    def __init__(self):
        self.NODE_NAME = 'SimpleTextImage'

    @classmethod
    def INPUT_TYPES(self):

        (_, FONT_DICT) = get_resource_dir()
        FONT_LIST = list(FONT_DICT.keys())

        return {
            "required": {
                "text": ("STRING",{"default": "text", "multiline": True},
                ),
                "font_file": (FONT_LIST,),
                "align": (["center", "left", "right"],),
                "char_per_line": ("INT", {"default": 80, "min": 1, "max": 8096, "step": 1},),
                "leading": ("INT",{"default": 8, "min": 0, "max": 8096, "step": 1},),
                "font_size": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "text_color": ("STRING", {"default": "#FFFFFF"},),
                "stroke_width": ("INT",{"default": 0, "min": 0, "max": 8096, "step": 1},),
                "stroke_color": ("STRING",{"default": "#FF8000"},),
                "x_offset": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "y_offset": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "width": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "height": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'simple_text_image'
    CATEGORY = '😺dzNodes/LayerUtility'
    

    def simple_text_image(self, text, font_file, align, char_per_line,
                          leading, font_size, text_color,
                          stroke_width, stroke_color, x_offset, y_offset,
                          width, height, size_as=None
                          ):

        (_, FONT_DICT) = get_resource_dir()
        FONT_LIST = list(FONT_DICT.keys())

        ret_images = []
        ret_masks = []
        if size_as is not None:
            if size_as.dim() == 2:
                size_as_image = torch.unsqueeze(mask, 0)
            if size_as.shape[0] > 0:
                size_as_image = torch.unsqueeze(size_as[0], 0)
            else:
                size_as_image = copy.deepcopy(size_as)
            width, height = tensor2pil(size_as_image).size
        font_path = FONT_DICT.get(font_file)
        (_, top, _, _) = ImageFont.truetype(font=font_path, size=font_size, encoding='unic').getbbox(text)
        font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size))
        if char_per_line == 0:
            char_per_line = int(width / font_size)
        paragraphs = text.split('\n')

        img_height = height  # line_height * len(lines)
        img_width = width  # max(font.getsize(line)[0] for line in lines)

        img = Image.new("RGBA", size=(img_width, img_height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        y_text = y_offset + stroke_width
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, width=char_per_line, expand_tabs=False,
                                  replace_whitespace=False, drop_whitespace=False)
            for line in lines:
                width = font.getbbox(line)[2] - font.getbbox(line)[0]
                height = font.getbbox(line)[3] - font.getbbox(line)[1]
                # 根据 align 参数重新计算 x 坐标
                if align == "left":
                    x_text = x_offset
                elif align == "center":
                    x_text = (img_width - width) // 2
                elif align == "right":
                    x_text = img_width - width - x_offset
                else:
                    x_text = x_offset  # 默认为左对齐

                draw.text(
                    xy=(x_text, y_text),
                    text=line,
                    fill=text_color,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                    )
                y_text += height + leading
            y_text += leading * 2

        if size_as is not None:
            for i in size_as:
                ret_images.append(pil2tensor(img))
                ret_masks.append(image2mask(img.split()[3]))
        else:
            ret_images.append(pil2tensor(img))
            ret_masks.append(image2mask(img.split()[3]))

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        return (torch.cat(ret_images, dim=0),torch.cat(ret_masks, dim=0),)

class SimpleTextImageV2:

    def __init__(self):
        self.NODE_NAME = 'SimpleTextImageV2'

    @classmethod
    def INPUT_TYPES(self):

        (_, FONT_DICT) = get_resource_dir()
        FONT_LIST = list(FONT_DICT.keys())

        return {
            "required": {
                "texts": ("STRING",{"default": "text", "multiline": True},
                ),
                "font_file": (FONT_LIST,),
                "align": (["center", "left", "right"],),
                "char_per_line": ("INT", {"default": 80, "min": 1, "max": 8096, "step": 1},),
                "leading": ("INT",{"default": 8, "min": 0, "max": 8096, "step": 1},),
                "font_size": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "text_color": ("STRING", {"default": "#FFFFFF"},),
                "stroke_width": ("INT",{"default": 0, "min": 0, "max": 8096, "step": 1},),
                "stroke_color": ("STRING",{"default": "#FF8000"},),
                "x_offsets": ("STRING", {"default": 0, "min": 0, "max": 8096, "step": 1},),
                "y_offsets": ("STRING", {"default": 0, "min": 0, "max": 8096, "step": 1},),
                "widths": ("STRING", {"default": 512, "min": 1, "max": 8096, "step": 1},),
                "heights": ("STRING", {"default": 512, "min": 1, "max": 8096, "step": 1},),
                "img_width": ("STRING", {"default": 512, "min": 1, "max": 8096, "step": 1},),
                "img_height": ("STRING", {"default": 512, "min": 1, "max": 8096, "step": 1},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'simple_text_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'
    
    def simple_text_image_v2(self, texts, font_file, align, char_per_line,
                          leading, font_size, text_color,
                          stroke_width, stroke_color, x_offsets, y_offsets,
                          widths, heights, img_width, img_height, size_as=None
                          ):

        (_, FONT_DICT) = get_resource_dir()
        FONT_LIST = list(FONT_DICT.keys())

        ret_images = []
        ret_masks = []
        if size_as is not None:
            if size_as.dim() == 2:
                size_as_image = torch.unsqueeze(mask, 0)
            if size_as.shape[0] > 0:
                size_as_image = torch.unsqueeze(size_as[0], 0)
            else:
                size_as_image = copy.deepcopy(size_as)
            width, height = tensor2pil(size_as_image).size
        font_path = FONT_DICT.get(font_file)
        (_, top, _, _) = ImageFont.truetype(font=font_path, size=font_size, encoding='unic').getbbox(texts)
        font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size))
        if char_per_line == 0:
            char_per_line = int(width / font_size)
        paragraphs = texts.split('\n')
        print(f"paragraph:{paragraphs}")
        img = Image.new("RGBA", size=(img_width, img_height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        texts = texts.split("|")
        x_offsets = x_offsets.split("|")
        y_offsets = y_offsets.split("|")
        widths = widths.split("|")
        heights = heights.split("|")
        for idx, text in enumerate(texts):
            print("debug only  idx",idx,"text",text)
            x_offset = int(x_offsets[idx])
            y_offset = int(y_offsets[idx])
            width = int(widths[idx])
            height = int(heights[idx])

            # 计算字体大小比例
            font_size_ratio = max(1, min(100, int((height / img_height) * 100)))
            adjusted_font_size = int(font_size * (font_size_ratio / 70))
            font = ImageFont.truetype(font_path, adjusted_font_size)

            y_text = y_offset + stroke_width
            paragraphs = text.split('\n')
            for paragraph in paragraphs:
                lines = textwrap.wrap(paragraph, width=char_per_line, expand_tabs=False,
                                      replace_whitespace=False, drop_whitespace=False)
                for line in lines:
                    line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
                    line_height = font.getbbox(line)[3] - font.getbbox(line)[1]

                    # 根据 align 参数重新计算 x 坐标
                    if align == "left":
                        x_text = x_offset
                    elif align == "center":
                        x_text = x_offset + (width - line_width) // 2
                    elif align == "right":
                        x_text = x_offset + width - line_width
                    else:
                        x_text = x_offset  # 默认为左对齐

                    draw.text(
                        xy=(x_text, y_text),
                        text=line,
                        fill=text_color,
                        font=font,
                        stroke_width=stroke_width,
                        stroke_fill=stroke_color,
                    )
                    y_text += line_height + leading
                y_text += leading * 2
        if size_as is not None:
            for i in size_as:
               ret_images.append(pil2tensor(img))
               ret_masks.append(image2mask(img.split()[3]))
        else:
            ret_images.append(pil2tensor(img))
            ret_masks.append(image2mask(img.split()[3]))

        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)




NODE_CLASS_MAPPINGS = {
    "LayerUtility: SimpleTextImage": SimpleTextImage,
    "LayerUtility: SimpleTextImageV2": SimpleTextImageV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: SimpleTextImage": "LayerUtility: SimpleTextImage",
    "LayerUtility: SimpleTextImageV2": "LayerUtility: SimpleTextImageV2"
}

