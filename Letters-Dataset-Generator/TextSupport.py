from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

def has_glyph(font, glyph):
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def isTextSupported(font_directory, text):
    font = TTFont(font_directory)

    Found = all([has_glyph(font, char) for char in text])
    return Found