import os, re, time, json, threading, unicodedata
import cv2
import numpy as np
import pandas as pd

# UTF-8 pentru terminal pe Windows (evită UnicodeEncodeError la diacritice)
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
 
CARD_W, CARD_H = 1014, 640
EXCEL_FILE     = "baza_date_buletine.xlsx"
JSON_FILE      = "rezultate.json"
ASSETS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ──────────────────────────────────────────────────────────────────────────────
# 1. DETECTARE CARD & CORECȚIE UNGHI
# ──────────────────────────────────────────────────────────────────────────────

def _order_pts(pts):
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _detect_card_by_color(img):
    """
    Detectează cardul prin culoarea sa caracteristică: albastru-deschis/cyan
    (specifică buletinului românesc) folosind mascare HSV.

    Returnează masca binară (uint8, 0/255).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Intervalul HSV pentru albastru-cyan deschis al buletinului:
    # H: 85–135, S: 15–200, V: 120–255 (range larg pentru condiții variate de lumină)
    mask = cv2.inRange(hsv,
                       np.array([85,  15, 120]),
                       np.array([135, 200, 255]))

    # Morfologie: umple găuri și elimină zgomot mic
    k_close = np.ones((25, 25), np.uint8)
    k_open  = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    return mask


def _fix_warp_orientation(warped: np.ndarray, method: str):
    """
    Verifică dacă warp-ul a plasat conținutul sideways sau cu capul în jos.
    Folosește banner-ul albastru 'ROMANIA' ca indicator principal.
    """
    h, w = warped.shape[:2]
    
    # 1. Verificare prin Culoare (ROMANIA banner e albastru)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    # Albastru: Hue 100-130, Sat 100-255, Val 50-255 (ajustat pt foto)
    lower_blue = np.array([100, 50, 40])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    margin_h = h // 5
    margin_w = w // 5
    
    top_blue    = np.sum(blue_mask[:margin_h, :] > 0)
    bottom_blue = np.sum(blue_mask[h - margin_h:, :] > 0)
    left_blue   = np.sum(blue_mask[:, :margin_w] > 0)
    right_blue  = np.sum(blue_mask[:, w - margin_w:] > 0)
    
    blue_scores = {'top': top_blue, 'bottom': bottom_blue, 'left': left_blue, 'right': right_blue}
    dominant_blue = max(blue_scores, key=blue_scores.get)
    
    # Dacă avem o detecție albastră clară, decidem pe baza ei
    if blue_scores[dominant_blue] > 500:
        if dominant_blue == 'top':
            return warped, method
        elif dominant_blue == 'bottom':
            return cv2.rotate(warped, cv2.ROTATE_180), method + "+rot180"
        elif dominant_blue == 'left':
            fixed = cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), (CARD_W, CARD_H), cv2.INTER_CUBIC)
            return fixed, method + "+rot90cw"
        else:
            fixed = cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE), (CARD_W, CARD_H), cv2.INTER_CUBIC)
            return fixed, method + "+rot90ccw"

    # 2. Fallback: Intensitate (cea mai neagră margine e header-ul)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    inv  = 255 - gray
    top    = float(inv[:margin_h].mean())
    bottom = float(inv[h - margin_h:].mean())
    left   = float(inv[:, :margin_w].mean())
    right  = float(inv[:, w - margin_w:].mean())

    dominant = max({'top': top, 'left': left, 'right': right, 'bottom': bottom}, key=lambda k: {'top': top, 'left': left, 'right': right, 'bottom': bottom}[k])

    if dominant == 'top': return warped, method
    elif dominant == 'bottom': return cv2.rotate(warped, cv2.ROTATE_180), method + "+rot180"
    elif dominant == 'left':
        fixed = cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), (CARD_W, CARD_H), cv2.INTER_CUBIC)
        return fixed, method + "+rot90cw"
    else:
        fixed = cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE), (CARD_W, CARD_H), cv2.INTER_CUBIC)
        return fixed, method + "+rot90ccw"




def _validate_warp(warped: np.ndarray, original: np.ndarray) -> bool:
    """
    Validare post-warp: verifică dacă perspectiva a produs o imagine de calitate acceptabilă.
    Compară sharpness-ul, contrastul și densitatea marginilor cu imaginea originală.
    Returnează False dacă warp-ul pare o detecție falsă (se va folosi originalul).
    """
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness: Laplacian variance – warp-ul bun păstrează claritatea textului
    lap_w = cv2.Laplacian(gray_w, cv2.CV_64F).var()
    lap_o = cv2.Laplacian(gray_o, cv2.CV_64F).var()
    if lap_o > 0 and lap_w < lap_o * 0.12:
        return False   # warp mult mai blur decât originalul → detecție falsă

    # 2. Contrast: std deviation – imagine aproape monocromatică = probabil fundal
    if gray_w.std() < 18:
        return False

    # 3. Densitate margini în zona text (treimea inferioară = zona adresă/date)
    h = gray_w.shape[0]
    edges = cv2.Canny(gray_w[h // 3:, :], 50, 150)
    edge_density = edges.sum() / (edges.size * 255.0)
    if edge_density < 0.004:   # prea puține margini → nu e text real
        return False

    # 4. Banner albastru ROMANIA în zona superioară (cea mai specifică verificare)
    # Un warp valid are bannerul caracteristic albastru în top ~20% după orientare.
    # O detecție falsă (fundal, alt obiect) nu îl va avea.
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_w,
                            np.array([100,  50,  40]),
                            np.array([135, 255, 255]))
    top_blue = int(np.sum(blue_mask[:h // 5, :] > 0))
    if top_blue < 1000:
        return False   # fără banner albastru sus → nu e card valid

    return True



def deskew_and_crop(img, _depth=0, skip_warp=False):
    """
    Încearcă să detecteze cardul și să corecteze perspectiva/unghiul.

    Strategii (în cascadă):
      0. HSV color masking → contur cu 4 laturi → warp  (fundal texturat)
      1. Canny → contur cu 4 laturi → warp perspectivă  (fundal cu contrast)
      2. Canny → minAreaRect → rotație simplă
      3. Original (nicio modificare)

    skip_warp=True sare Strategiile 0 și 1 (warp perspectivă) și începe direct cu
    rotația simplă (Strategy 2). Folosit când am confirmat că warp-ul produce garbage.
    Returnează (img_procesat, metoda_string).
    """
    img_area   = img.shape[0] * img.shape[1]
    _cnts_area = img_area   # area imaginii din care vin cnts (poate fi downsamplat)

    if not skip_warp:
        # ── Strategia 0: HSV color masking ──
        # Funcționează bine când fundalul NU are aceeași culoare cu cardul.
        mask = _detect_card_by_color(img)
        cnts_c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_c = sorted(cnts_c, key=cv2.contourArea, reverse=True)
        for c in cnts_c[:4]:
            if cv2.contourArea(c) < img_area * 0.05:
                break
            peri   = cv2.arcLength(c, True)
            # Încercăm mai multe toleranțe eps pentru approxPolyDP
            for eps in [0.02, 0.03, 0.04, 0.05, 0.08]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4:
                    pts = _order_pts(approx.reshape(4, 2))
                    # Fix: verificăm aspect ratio-ul quad-ului (buletinul e ~1.58:1)
                    w_est = np.linalg.norm(pts[1] - pts[0])
                    h_est = np.linalg.norm(pts[3] - pts[0])
                    ratio = max(w_est, h_est) / (min(w_est, h_est) or 1)
                    if not (1.3 < ratio < 1.9):
                        continue
                    dst = np.float32([[0, 0], [CARD_W-1, 0],
                                      [CARD_W-1, CARD_H-1], [0, CARD_H-1]])
                    M = cv2.getPerspectiveTransform(pts, dst)
                    warped = cv2.warpPerspective(img, M, (CARD_W, CARD_H))
                    result_w, method_w = _fix_warp_orientation(warped, "warp(color)")
                    if _validate_warp(result_w, img):
                        return result_w, method_w
                    # Warp de calitate slabă – continuăm cu strategiă următoare

        # ── Strategia 1: Canny edges → warp 4 puncte ──
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts     = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts[:6]:
            if cv2.contourArea(c) < img_area * 0.05:
                break
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = _order_pts(approx.reshape(4, 2))
                dst = np.float32([[0, 0], [CARD_W-1, 0], [CARD_W-1, CARD_H-1], [0, CARD_H-1]])
                M   = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(img, M, (CARD_W, CARD_H))
                result_w, method_w = _fix_warp_orientation(warped, "warp")
                if _validate_warp(result_w, img):
                    return result_w, method_w
                # Warp Canny slab – continuăm cu strategia de rotație
    else:
        # skip_warp=True: calculăm Canny pe imagine downsamplată (max 1500px)
        # Canny pe imagini 4K produce contururi fragmentate → downscale înainte.
        _hs, _ws = img.shape[:2]
        _scale   = min(1.0, 1500.0 / max(_hs, _ws))
        img_ds   = cv2.resize(img, (int(_ws * _scale), int(_hs * _scale))) if _scale < 1.0 else img
        gray  = cv2.cvtColor(img_ds, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts     = sorted(cnts, key=cv2.contourArea, reverse=True)
        _cnts_area = img_ds.shape[0] * img_ds.shape[1]

    # ── Strategia 2: rotație simplă (EXTREME CAUTION) ──
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        rect  = cv2.minAreaRect(c)
        (center, (w_r, h_r), angle) = rect

        # Verificăm aspect ratio-ul cutiei (buletinul e ~1.58)
        ratio = max(w_r, h_r) / (min(w_r, h_r) or 1.0)

        if angle < -45: angle += 90
        elif angle > 45: angle -= 90

        # Decizie de rotație:
        # - Dacă imaginea e deja landscape (w > h) și unghiul e mare (> 8°),
        #   și cardul nu ocupă TOATĂ imaginea, probabil e detectat greșit (ex. diagonală de fundal).
        h_orig, w_orig = img.shape[:2]
        is_landscape = w_orig > h_orig * 1.1
        # _cnts_area: area imaginii din care provin cnts (poate fi img_ds dacă skip_warp)
        _c_area = cv2.contourArea(c)
        
        should_rotate = False
        if 1.2 < ratio < 1.9:
            if is_landscape:
                if abs(angle) < 6.0:
                    should_rotate = True
                elif abs(angle) <= 25.0 and 1.2 < ratio < 2.0:
                    # Card la unghi moderat (6-25°) cu raport aspect card-like
                    should_rotate = True
                elif _c_area > _cnts_area * 0.8:
                    should_rotate = True
            else:
                if abs(angle) > 2.0: should_rotate = True

        if should_rotate:
            h, w = img.shape[:2]
            M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            # Fix P10: doar pt carduri foarte clare încercăm fix orientation în Strategy 2
            if _c_area > _cnts_area * 0.35:
                return _fix_warp_orientation(rotated, f"rotate({angle:.1f}°)")
            return rotated, f"rotate({angle:.1f}°)"

    # ── Strategia 4: portret → landscape (o singură recursie) ──
    # Cardul românesc e mereu landscape (~1014×640).
    # Dacă imaginea rămasă după strategiile 0-2 e portret, rotim 90° și rerulăm.
    if _depth == 0:
        h_img, w_img = img.shape[:2]
        if h_img > w_img * 1.1:
            cw  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Alegăm sensul cu mai mult albastru în jumătatea superioară:
            # buletinul are banda albastră la STEG, deci sus când e drept.
            def _upper_blue(i):
                m   = _detect_card_by_color(i)
                h_  = m.shape[0]
                tot = float(m.sum()) or 1.0
                return m[:h_ // 2].sum() / tot
            if _upper_blue(cw) >= _upper_blue(ccw):
                best, label = cw,  "rot90cw"
            else:
                best, label = ccw, "rot90ccw"
            sub_result, sub_method = deskew_and_crop(best, _depth=1)
            return sub_result, f"{label}+{sub_method}"

    return img, "original"


# ──────────────────────────────────────────────────────────────────────────────
# 2. RESTAURARE DIACRITICE PENTRU NUME PROPRII
# ──────────────────────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    """Elimină diacriticele pentru comparație normalizată."""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Prenume românești frecvente → formă cu diacritice corecte.
# Cheia = același token fără diacritice, majuscule.
_PRENUME_DICT: dict[str, str] = {
    "LAURENTIU": "LAURENȚIU", "FLORENTIU": "FLORENȚIU",
    "VINCENTIU": "VINCENȚIU",  "DORENTIU":  "DORENȚIU",
    "IONUT":     "IONUȚ",      "PETRUT":    "PETRUȚ",
    "CATALIN":   "CĂTĂLIN",    "CATALINA":  "CĂTĂLINA",
    "STEFAN":    "ȘTEFAN",     "STEFANIA":  "ȘTEFANIA",
    "MADALINA":  "MĂDĂLINA",   "RAZVAN":    "RĂZVAN",
    "TANASE":    "TĂNASE",     "CALIN":     "CĂLIN",
    "CALINA":    "CĂLINA",     "CALINA":    "CĂLINA",
    "MIHAITA":   "MIHĂIȚĂ",    "GHEORGHITA":"GHEORGHIȚĂ",
    "LUMINITA":  "LUMINIȚA",   "ANITA":     "ANIȚĂ",
    "CONSTANTA": "CONSTANȚA",  "MARIUTA":   "MĂRIUȚĂ",
    "STELUTA":   "STELUȚA",    "MARITA":    "MARIȚA",
    "DRAGOS":    "DRAGOȘ",     "COSTEL":    "COSTEL",
}
# Lookup normalizat (fără diacritice, majuscule)
_PRENUME_LOOKUP: dict[str, str] = {
    _strip_accents(k).upper(): v for k, v in _PRENUME_DICT.items()
}


def _apply_diacritic_rules(word: str) -> str:
    """
    Aplică reguli lingvistice românești pentru reconstituirea diacriticelor
    unui cuvânt (nume propriu sau localitate) dat în ASCII uppercase.

    Acoperă:
      - Prefixe:  T→Ț/Ș la început urmat de pattern specific
      - Interioare: secvențe interne caracteristice românei
      - Sufixe: terminații morfologice frecvente
    """
    p = word.upper().strip()
    if not p:
        return p

    # ── Prefixe ──
    p = re.sub(r'^TIN',   'ȚIN',   p)   
    p = re.sub(r'^TIR',   'ȚIR',   p)   
    p = re.sub(r'^TIG',   'ȚIG',   p)   
    p = re.sub(r'^TIP',   'ȚIP',   p)
    p = re.sub(r'^STE',   'ȘTE',   p)   
    p = re.sub(r'^STI',   'ȘTI',   p)  
    p = re.sub(r'^SARA',  'ȘARA',  p)
    p = re.sub(r'^SERB',  'ȘERB',  p)
    p = re.sub(r'^SIRBU', 'ȘIRBU', p)

    # ── Interioare ──
    # Interior: IN/IR între consoane (incluzând Ț și Ș produse de regulile de prefix)
    p = re.sub(r'(?<=[BCDFGHJKLMNPQRSTVWXYZȚȘ])IN(?=[BCDFGHJKLMNPQRSTVWXYZȚȘ])', 'ÎN', p)
    p = re.sub(r'(?<=[BCDFGHJKLMNPQRSTVWXYZȚȘ])IR(?=[BCDFGHJKLMNPQRSTVWXYZȚȘ])', 'ÎR', p)
    # Ț interior (vocală+T+vocală — ex: PIATA→PIAȚA)
    p = re.sub(r'(?<=[AEI])T(?=[AEI])',   'Ț',  p)

    # ── Sufixe ──
    p = re.sub(r'ESTI$',   'EȘTI',   p)  # ONESTI→ONEȘȚI, BRANESTI→BRĂNEȘTI
    p = re.sub(r'ESTI\b',  'EȘTI',   p)
    p = re.sub(r'ESTI ',   'EȘTI ',  p)
    p = re.sub(r'ESCU$',   'ESCU',   p)  
    p = re.sub(r'ENTIU$',  'ENȚIU',  p)  # LAURENTIU→LAURENȚIU
    p = re.sub(r'([BCDFGHJKLMNPQRSTVWXYZ])UTA$', r'\1UȚĂ', p)  # STELUTA→STELUȚĂ
    p = re.sub(r'NITA$',   'NIȚĂ',   p)  # LUMINITA→LUMINIȚA
    p = re.sub(r'([BCDFGHJKLMNPQRSTVWXYZ])UT$',  r'\1UȚ',  p)  # IONUT→IONUȚ
    p = re.sub(r'ARA$',    'ARĂ',    p)  # TARA→ȚARĂ, SARA→ȘARĂ
    p = re.sub(r'ASCA$',   'ASCĂ',   p)  # ROMANEASCA→ROMÂNEASCĂ
    p = re.sub(r'EASCA$',  'EASCĂ',  p)

    return p


def _restore_localitate_rules(loc: str) -> str:
    """
    Aplică reguli specifice pentru localități: sufixe și prefixe frecvente
    în toponomia românească.
    """
    p = loc.upper().strip()
    # Sufixe foarte frecvente în localități
    p = re.sub(r'ESTI$',   'EȘTI',  p)   # Onești, Brănești, Cornești
    p = re.sub(r'ESTI\b',  'EȘTI',  p)
    p = re.sub(r'ASCA$',   'ASCĂ',  p)   # Dorobaneasca
    p = re.sub(r'OARA$',   'OARĂ',  p)   # Timișoara, Câmpulung Moldoveneasca
    p = re.sub(r'ARA$',    'ARĂ',   p)   # nu e frecvent în localități, lasăm
    # Prefixe
    p = re.sub(r'^TIRG',   'TÂRG',  p)   # TIRGU→TÂRGU
    p = re.sub(r'^TARGU',  'TÂRGU', p)   # TARGU MURES→TÂRGU MUREȘ
    p = re.sub(r'^RAMNICU','RÂMNICU',p)  # RAMNICU→RÂMNICU
    p = re.sub(r'^CAMPUL', 'CÂMPUL',p)
    p = re.sub(r'^CIMPU',  'CÂMPU', p)
    # Interior â/î
    p = re.sub(r'(?<=[BCDFGHJKLMNPQRSTVWXYZ])IN(?=[BCDFGHJKLMNPQRSTVWXYZ])',
               'ÎN', p)
    # Mureș, Iași etc.
    p = re.sub(r'MURES$',  'MUREȘ', p)
    p = re.sub(r'IASI$',   'IAȘI',  p)
    p = re.sub(r'GALATI$', 'GALAȚI',p)
    p = re.sub(r'BACAU$',  'BACĂU', p)
    p = re.sub(r'BRASOV$', 'BRAȘOV',p)
    p = re.sub(r'CONSTANTA$','CONSTANȚA',p)
    p = re.sub(r'PLOIESTI$','PLOIEȘTI',p)
    p = re.sub(r'TIMISOARA$','TIMIȘOARA',p)
    p = re.sub(r'ONESTI$', 'ONEȘTI', p)
    p = re.sub(r'ZARNESTI$','ZĂRNEȘTI',p)
    p = re.sub(r'FAGARAS$', 'FĂGĂRAȘ',p)
    p = re.sub(r'BISTRITA$','BISTRIȚA',p)
    p = re.sub(r'BOTOSANI$','BOTOȘANI',p)
    p = re.sub(r'BUZAU$',   'BUZĂU',  p)
    p = re.sub(r'PITESTI$', 'PITEȘTI', p)
    p = re.sub(r'TARGOVISTE$','TÂRGOVIȘTE',p)
    p = re.sub(r'SIGHISOARA$','SIGHIȘOARA',p)
    p = re.sub(r'MEDIAS$',  'MEDIAȘ', p)
    p = re.sub(r'ZALAU$',   'ZALĂU',  p)
    p = re.sub(r'RESITA$',  'REȘIȚA', p)
    p = re.sub(r'SUCEAVA$', 'Suceava',p)
    return p.title()


def restore_prenume_diacritics(prenume: str) -> str:
    """Restaurează diacriticele unui prenume (simplu sau compus cu '-')."""
    parts = prenume.split('-')
    out = []
    for part in parts:
        key = _strip_accents(part.upper())
        if key in _PRENUME_LOOKUP:
            out.append(_PRENUME_LOOKUP[key])
        else:
            out.append(_apply_diacritic_rules(part))
    return '-'.join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESARE (grayscale + zoom + contrast ușor)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(img_bgr, min_width=2200):
    """
    Grayscale → zoom in dacă e prea mică → sharpening → CLAHE (egalizare locală contrast).
    Returnează imaginea grayscale pregătită pentru OCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1800:
        scale = min_width / w
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)
        print(f"[INFO] Zoom in: {w}→{int(w*scale)} px lățime")
    elif w > 3500:
        scale = 2500 / w
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
        print(f"[INFO] Scale down: {w}→{int(w*scale)} px lățime")

    # Sharpening ușor: ajută imaginile ușor blur (telefon, scanner slab)
    kernel_sharp = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel_sharp)

    # CLAHE – contrast local ușor, ajută la text slab imprimat
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EXTRAGERE CÂMPURI
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 3b. DICȚIONAR JUDEȚE & SEGMENTARE ADRESĂ
# ──────────────────────────────────────────────────────────────────────────────

# Abreviere (2-3 litere, cf. plăcuțelor auto + standard românesc) → Nume complet
JUDETE: dict[str, str] = {
    "AB": "Alba",
    "AR": "Arad",
    "AG": "Argeș",
    "BC": "Bacău",
    "BH": "Bihor",
    "BN": "Bistrița-Năsăud",
    "BT": "Botoșani",
    "BV": "Brașov",
    "BR": "Brăila",
    "B":  "București",
    "BIF": "Ilfov",       # Ilfov apare uneori ca IF sau BIF pe documente
    "IF": "Ilfov",
    "BZ": "Buzău",
    "CS": "Caraș-Severin",
    "CL": "Călărași",
    "CJ": "Cluj",
    "CT": "Constanța",
    "CV": "Covasna",
    "DB": "Dâmbovița",
    "DJ": "Dolj",
    "GL": "Galați",
    "GR": "Giurgiu",
    "GJ": "Gorj",
    "HR": "Harghita",
    "HD": "Hunedoara",
    "IL": "Ialomița",
    "IS": "Iași",
    "MM": "Maramureș",
    "MH": "Mehedinți",
    "MS": "Mureș",
    "NT": "Neamț",
    "OT": "Olt",
    "PH": "Prahova",
    "SM": "Satu Mare",
    "SJ": "Sălaj",
    "SB": "Sibiu",
    "SV": "Suceava",
    "TR": "Teleorman",
    "TM": "Timiș",
    "TL": "Tulcea",
    "VS": "Vaslui",
    "VL": "Vâlcea",
    "VN": "Vrancea",
}

# Codul județului din CNP (cifrele 8-9, 1-indexed) → Abrevierea județului
# Sursa: Ord. MAI privind CNP + tabel oficial ANAF
CNP_JUDETE: dict[str, str] = {
    "01": "AB", "02": "AR", "03": "AG", "04": "BC", "05": "BH",
    "06": "BN", "07": "BT", "08": "BV", "09": "BR", "10": "BZ",
    "11": "CS", "12": "CJ", "13": "CT", "14": "CV", "15": "DB",
    "16": "DJ", "17": "GL", "18": "GJ", "19": "HR", "20": "HD",
    "21": "IL", "22": "IS", "23": "IF", "24": "MM", "25": "MH",
    "26": "MS", "27": "NT", "28": "OT", "29": "PH", "30": "SM",
    "31": "SJ", "32": "SB", "33": "SV", "34": "TR", "35": "TM",
    "36": "TL", "37": "VS", "38": "VL", "39": "VN", "40": "B",
    "41": "B",  "42": "B",  "43": "B",  "44": "B",  "45": "B",
    "46": "B",                               # Sectoarele 1-6 București
    "51": "XX", "52": "XX",                  # străini cu reşedință în RO
}


def cnp_cod_judet(cnp: str) -> str:
    """
    Extrage câmpul Cod_judet din CNP (cifrele 8-9, 1-indexed).
    Returnează codul numeric 2 cifre (ex. "29" pentru Prahova),
    sau "Nedetectat" dacă CNP-ul nu are forma corectă.
    """
    if cnp and len(cnp) == 13 and cnp.isdigit():
        return cnp[7:9]   # index 7-8 (0-based) = cifra 8-9 (1-based)
    return "Nedetectat"

# Cuvinte-cheie pentru tipul localității (ordine: cel mai specific primul)
# Fix P1: variante cu și fără spațiu după punct (Mun.Onești, Orș.Ghimbav)
_LOC_KEYWORDS = [
    (r'\bMUN[.:]*(?:\s+|(?=[A-ZĂÂÎȘȚ]))', "MUN."),   # MUN. / MUN: / MUN.Onesti
    (r'\bMUNICIPIUL?\s+',                "MUN."),
    (r'\bOR[SȘ][.:]?(?:\s+|(?=[A-ZĂÂÎȘȚ]))', "OR."),   # ORȘ. / ORȘ: / ORȘ.Ghimbav
    (r'\bOR\.\s+',                      "OR."),
    (r'\bORA[ȘS]UL?\s+',               "OR."),
    (r'\bCOM[.:]?(?:\s+|(?=[A-ZĂÂÎȘȚ]))',  "COM."),  # COM. / COM: / COM.Dumbrava
    (r'\bCOMUNA?\s+',                   "COM."),
    (r'\bSAT[.:]?(?:\s+|(?=[A-ZĂÂÎȘȚ]))',  "SAT"),   # SAT. / SAT: / SAT.Negoiesti
    (r'\bSECT?\.?\s+',                  "SECT."),
    (r'\bSECTORUL?\s+',                 "SECT."),
    (r'\bSEC\.\s+',                     "SECT."),
]

# Cuvinte-cheie pentru stradă.
# Acoperă: STR POPA (fără punct), STR. POPA (cu punct + spațiu),
# STR.Podul (cu punct fără spațiu) – toate variantele OCR frecvente.
_STR_KEYWORDS = [
    r'\bSTR\.?(?:\s+|(?=[A-ZĂÂÎȘȚ]))',  # STR / STR. / STR.Podul
    r'\bSTRADA?\b\s*',
    r'\bBD\.?(?:\s+|(?=[A-ZĂÂÎȘȚ]))',   # BD / BD.  / BD.Unirii
    r'\bBDUL\.?\s+',
    r'\bBULEVARDUL?\s+', r'\bCALEA\s+', r'\bALEEA\s+',
    r'\bINTRAREA\s+', r'\bSPLAIUL?\s+', r'\bPIAȚA?\s+',
    r'\bSOSEAUA?\s+', r'\bDRUMUL?\s+',
    r'\bSOS\.?\s+',          # SOS. / SOS (abreviere Șoseaua)
    r'\b[ȘS]OS\.?\s+',       # ȘOS. cu diacritică
    r'\bP-?ȚA\.?\s+',        # P-ȚA / PȚA (Piața abreviat)
    r'\bCAL\.?\s+',          # CAL. (Calea abreviat)
    r'\bSPL\.?\s+',          # SPL. (Splaiul abreviat)
]


def parse_address(adresa: str) -> dict[str, str]:
    """
    Segmentează adresa brută în:
      - Judet:     numele complet al județului (PH → Prahova)
      - Localitate: tipul + numele localității (MUN. Ploiești / SAT Florești etc.)
      - Strada:    tot ce urmează după marcatorul de stradă (STR., BD. etc.)

    Returnează dict cu cheile Judet, Localitate, Strada.
    Câmpurile ne-detectate rămân "Nedetectat".
    """
    result = {"Judet": "Nedetectat", "Localitate": "Nedetectat", "Strada": "Nedetectat"}
    text = adresa.upper().strip()

    # ── 1. Județ ──
    # Normalizare JUD: "JUD BVHUN" / "JUDBVHUN" / "JUD.BVMUN" → "JUD BV MUN"
    # HUN și MUN sunt corupții OCR ale lui "Mun." (municipiu)
    text = re.sub(
        r'\bJUD\.?\s*([A-Z]{2,3})\s*(MUN|HUN)\b',
        r'JUD \1 MUN', text, flags=re.IGNORECASE
    )
    # Formă: JUD. PH  /  JUD PH  /  JUDEȚUL PH  /  JUD TELEORMAN (nume complet)
    m_jud = re.search(
        r'\bJUD(?:E[TȚ](?:UL)?)?[.:]*\s*([A-ZĂÂÎȘȚ]{2,15})\b',
        text
    )
    # Caz special: adrese București fără prefix JUD. (ex. "MUN BUCURESTI SEC. 4")
    if not m_jud:
        if re.search(r'\bBUCURE[ȘS]TI?\b', text):
            result["Judet"] = "București"
            judet_abr = "B"
        else:
            judet_abr = ""
    else:
        judet_abr = ""
    if m_jud:
        raw = m_jud.group(1).strip().rstrip('.')
        # Dacă e deja numele complet (ex. PRAHOVA), îl căutăm invers în dicționar
        if raw in JUDETE:
            result["Judet"] = JUDETE[raw]
            judet_abr = raw
        else:
            # Caută după nume complet (normalizat fără diacritice)
            for abr, nume in JUDETE.items():
                if _strip_accents(nume).upper() == _strip_accents(raw).upper():
                    result["Judet"] = nume
                    judet_abr = abr
                    break
            if result["Judet"] == "Nedetectat":
                # Păstrăm ce a găsit OCR-ul, cel puțin
                result["Judet"] = raw.capitalize()

    # ── 2. Localitate ──
    # Tăiem prefixul JUD.XX din text pentru a nu-l confunda cu localitatea
    text_no_jud = re.sub(
        r'\bJUD(?:E[TȚ](?:UL)?)?[.:]*\s*[A-ZĂÂÎȘȚ]{1,15}\b[,\s]*', '', text
    ).strip()

    loc_start = None
    loc_match_end = None
    for pattern, prefix in _LOC_KEYWORDS:
        m = re.search(pattern, text_no_jud)
        if m:
            if loc_start is None or m.start() < loc_start:
                loc_start     = m.start()
                loc_match_end = m.end()   # capătul match-ului (după spațiu)

    if loc_match_end is not None:
        after_loc = text_no_jud[loc_match_end:].lstrip('. ').strip()
    else:
        after_loc = text_no_jud  # fallback: tot textul rămas

    # Găsim marcatorul de stradă în after_loc
    str_start = None
    for pat in _STR_KEYWORDS:
        m = re.search(pat, after_loc)
        if m:
            if str_start is None or m.start() < str_start:
                str_start = m.start()

    # Localitate = textul de la după prefix până la stradă (sau virgulă)
    if str_start is not None:
        loc_name_raw = after_loc[:str_start].strip().rstrip(',').strip()
    else:
        # Fără stradă; poate fi o virgulă separator
        parts = re.split(r'[,;]', after_loc, maxsplit=1)
        loc_name_raw = parts[0].strip()

    # Curățăm numărul poștal de la final, dacă există
    loc_name_raw = re.sub(r'\s+\d{4,6}\s*$', '', loc_name_raw).strip()

    # ── Curățare junk OCR interpolat în localitate ──
    # Cazul tipic: "BUCURESTI SEC 4 841 E4 B S.P.CEP . SECTOR 4"
    # → tokenii "841", "E4", "B", "S.P.CEP . SECTOR 4" nu fac parte din localitate
    # Strategia: localitate = secvența INIȚIALĂ de cuvinte valide (litere + SEC/SECT opțional cu cifre)
    # până la primul token de junk: cifre standalone, literă singură, cod alfanum scurt, emitent
    _EMITENT_RE = re.compile(
        r'\b(?:S\.?P\.?C\.?E\.?P\.?|SPCLEP|POLITIA|POLI[TȚ]IA|EUP|EVO)'
        r'(?:\s*\.?\s*\w+){0,3}',
        re.IGNORECASE
    )
    # Tăiem tot ce apare de la primul emitent/junk serios
    m_emit = _EMITENT_RE.search(loc_name_raw)
    if m_emit:
        loc_name_raw = loc_name_raw[:m_emit.start()].strip().rstrip(',').strip()

    # Eliminăm la final: cifre standalone, litere singure, coduri alfanumerice scurte (ex. "841", "E4", "B")
    # dar protejăm "SEC 4" / "SECTOR 3" la final (număr de sector valid)
    for _ in range(3):
        # Ștergem trailing letter+digit codes (E4, B5 etc.) și litere singure (B)
        loc_name_raw = re.sub(r'\s+[A-Z]\d+\s*$', '', loc_name_raw, flags=re.IGNORECASE).strip()
        loc_name_raw = re.sub(r'\s+[A-Z]\s*$', '', loc_name_raw, flags=re.IGNORECASE).strip()
        # Ștergem cifre standalone DOAR dacă nu sunt precedate de SEC/SECTOR
        if not re.search(r'\bSECT?(?:OR)?\s+\d+\s*$', loc_name_raw, re.IGNORECASE):
            loc_name_raw = re.sub(r'\s+\d+\s*$', '', loc_name_raw, flags=re.IGNORECASE).strip()

    # Eliminăm zgomotul de la început: cifre, litere singure, coduri reziduale
    loc_name_raw = re.sub(r'^(?:[A-Z]|\d+)\s+', '', loc_name_raw, flags=re.IGNORECASE).strip()
    loc_name_raw = re.sub(r'^(?:[A-Z]|\d+)\s+', '', loc_name_raw, flags=re.IGNORECASE).strip()

    # Localitate = DOAR numele localității (fără prefix MUN./SAT./etc.)
    if loc_name_raw and len(loc_name_raw) >= 2:
        # Fix C3: eliminăm trailing cifre standalone + abrevieri județe din localitate
        for _ in range(3):
            loc_name_raw = re.sub(r'\s+\d{2,6}\s*$', '', loc_name_raw).strip()
            loc_name_raw = re.sub(r'\s+[A-Z]{2,3}\s*$', '', loc_name_raw, flags=re.IGNORECASE).strip()
        # .title() capitalizează fiecare cuvânt (inclusiv cel din paranteză)
        loc_fmt = loc_name_raw.title()
        # OCR citește punctul dintre tip și comună ca spațiu sau cratimă:
        # "(Com-Dumbrava)" și "(Com Dumbrava)" → "(Com.Dumbrava)"
        loc_fmt = re.sub(r'\(Com[-\s]+', '(Com.', loc_fmt)
        loc_fmt = re.sub(r'\(Sat[-\s]+', '(Sat.', loc_fmt)
        result["Localitate"] = loc_fmt

    # ── 3. Stradă ──
    if str_start is not None:
        strada_raw = after_loc[str_start:].strip()
        # Curățăm codul poștal/index rezidual de la final (3-6 cifre standalone)
        strada_raw = re.sub(r'[,\s]+\d{3,6}\s*$', '', strada_raw).strip()
        strada_raw = re.sub(r'[,\s]+[A-Z]{2,3}\s*$', '', strada_raw).strip()
        # Fix C2: eliminăm emitentul SPCLEP/POLITIA din stradă
        strada_raw = re.sub(
            r'\s*(?:SPCLEP|POLI[TȚ]IA|EMIS[AĂ]?)\b.*$', '',
            strada_raw, flags=re.IGNORECASE
        ).strip()
        # Fix C5: normalizare Nr, → Nr. (virgulă OCR în loc de punct)
        strada_raw = re.sub(r'\bNr\s*,\s*', 'Nr. ', strada_raw, flags=re.IGNORECASE)
        # Normalizare număr: Nr: / NR : / nr: / nr 34 / NR 34 → Nr. → ex. Nr. 18
        strada_raw = re.sub(r'\bNr\.?\s*[:.]?\s*(?=\d)', 'Nr. ', strada_raw, flags=re.IGNORECASE)
        # Eliminăm prefixul tipului de stradă (Str., Bd., Calea, Aleea etc.)
        # → păstrăm DOAR numele străzii + numărul
        strada_raw = re.sub(
            r'^(?:STR\.?\s*|STRADA?\s+|BD\.?\s*|BDUL\.?\s+|BULEVARD(?:UL)?\s+'
            r'|CALEA\s+|CAL\.\s*|ALEEA\s+|INTRAREA\s+|SPLAIUL?\s+|SPL\.\s*|PIAȚA?\s+|P-?ȚA\.?\s*'
            r'|SOSEAUA?\s+|[ȘS]OS\.?\s*|DRUMUL?\s+)',
            '', strada_raw, flags=re.IGNORECASE
        ).strip()
        # Normalizare bl./sc./ap./et. → Bl. / Sc. / Ap. / Et. (Fix P5)
        for prefix in ('bl', 'sc', 'ap', 'et'):
            # Cazul 1: prefix + cifre (bl.12, bl12, bl 12)
            strada_raw = re.sub(
                rf'\b{prefix}\.?\s*(?=\d)', prefix.capitalize() + '. ',
                strada_raw, flags=re.IGNORECASE
            )
            # Cazul 2: prefix + identificator alfanumeric (ST6, A, C, A1, B12 etc.)
            # Acoperă: bLST6 → Bl. ST6, scC → Sc. C, sc.A → Sc. A
            strada_raw = re.sub(
                rf'\b{prefix}\.?\s*([A-Z][A-Z0-9]*)\b',
                lambda m, p=prefix: p.capitalize() + '. ' + m.group(1).upper(),
                strada_raw, flags=re.IGNORECASE
            )
        if len(strada_raw) >= 3:
            # Curățăm junk rezidual de la final: cifre/coduri standalone care NU fac parte din adresă
            # (ex. "075 BC" = index/stampilă de pe cardul fizic)
            def _strip_trailing_junk(s):
                _ADR_PFXS = re.compile(r'\b(?:nr|ap|bl|sc|et)\b', re.IGNORECASE)
                for _ in range(5):
                    # 2-3 litere standalone la final (cod județ: BC, CT, BV etc.)
                    new = re.sub(r'\s+[A-Z]{2,3}\s*$', '', s).strip()
                    if new != s:
                        s = new
                        continue
                    tokens = s.split()
                    if not tokens:
                        break
                    last = tokens[-1]
                    if re.fullmatch(r'\d{1,4}', last):
                        if len(tokens) >= 2:
                            prev = tokens[-2].rstrip('. ')  # strip punct/spațiu pentru comparație
                            if _ADR_PFXS.fullmatch(prev):
                                break  # cifra e legitimă (Nr. 100, Ap. 35 etc.)
                        s = ' '.join(tokens[:-1]).strip()
                    # Fix C6: coduri alfanumerice scurte (Ev2, E4, B5) care nu sunt prefixe de adresă
                    elif re.fullmatch(r'[A-Z]{1,2}\d{1,2}', last, re.IGNORECASE):
                        if not _ADR_PFXS.fullmatch(last):
                            s = ' '.join(tokens[:-1]).strip()
                    else:
                        break
                return s
            strada_raw = _strip_trailing_junk(strada_raw)
            strada_raw = strada_raw.title()
            # Re-uppercase identificatori alfanumerici după .title() (ex. St6 → ST6)
            strada_raw = re.sub(
                r'\b(Bl|Sc|Ap|Et)\.\s+([A-Z][a-z0-9]*[0-9][a-z0-9]*)\b',
                lambda m: m.group(1) + '. ' + m.group(2).upper(),
                strada_raw
            )
            result["Strada"] = strada_raw

    return result


def _sort_tokens_by_line(results, overlap_thresh=0.5):
    """
    Sortează tokenii pe linii bazat pe suprapunerea verticală (Y).
    Returnează o listă de tokeni (string) în ordinea citirii naturale.
    """
    if not results:
        return []

    # Fiecare element: [bbox, text, confidence]
    # Bbox: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    sorted_res = sorted(results, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)

    lines = []
    if sorted_res:
        current_line = [sorted_res[0]]
        for i in range(1, len(sorted_res)):
            prev = current_line[-1]
            curr = sorted_res[i]

            prev_y0, prev_y1 = prev[0][0][1], prev[0][2][1]
            curr_y0, curr_y1 = curr[0][0][1], curr[0][2][1]

            overlap = min(prev_y1, curr_y1) - max(prev_y0, curr_y0)
            h_min   = min(prev_y1 - prev_y0, curr_y1 - curr_y0)

            if overlap > h_min * overlap_thresh:
                current_line.append(curr)
            else:
                lines.append(current_line)
                current_line = [curr]
        lines.append(current_line)

    final_toks = []
    for line in lines:
        # Sortăm după X în cadrul liniei
        line.sort(key=lambda r: r[0][0][0])
        final_toks.extend([r[1].strip() for r in line if r[1].strip()])

    return final_toks


def extract_fields(results):
    """
    Din lista EasyOCR (detail=1), extrage câmpurile de interes.

    Metode:
      - Etichete spatiale (Nume, Prenume, Sex): cautăm eticheta, luăm tokenul următor
      - Regex pe textul complet (CNP, Serie, Numar, Adresă)
    """
    # Fix P8: Sortăm tokenele pe linii (mai robust la rotații mici decât ymid simplu)
    toks = _sort_tokens_by_line(results)

    # Fix C7: pre-normalizare tokeni lipiți (EasyOCR concatenează frecvent)
    # JudPH → Jud PH, OrșBăicoi → Orș Băicoi, etc.
    normalized_toks = []
    for t in toks:
        # Desparte JudXX (Jud + 2-3 litere majuscule = abreviere județ)
        t2 = re.sub(r'\b(Jud)([A-Z]{2,3})\b', r'\1 \2', t, flags=re.IGNORECASE)
        # Desparte Orș/Ors + Nume lipit (OrșBăicoi → Orș Băicoi)
        t2 = re.sub(r'\b(Or[șȘsS])\.?([A-ZĂÂÎȘȚ][a-zăâîșț]+)', r'\1 \2', t2)
        # Desparte Mun + Nume lipit
        t2 = re.sub(r'\b(Mun)\.?([A-ZĂÂÎȘȚ][a-zăâîșț]+)', r'\1 \2', t2, flags=re.IGNORECASE)
        # Desparte Loc + Nume lipit
        t2 = re.sub(r'\b(Loc)\.?([A-ZĂÂÎȘȚ][a-zăâîșț]+)', r'\1 \2', t2, flags=re.IGNORECASE)
        # Desparte Str + Nume lipit
        t2 = re.sub(r'\b(Str)\.?([A-ZĂÂÎȘȚ][a-zăâîșț]+)', r'\1 \2', t2, flags=re.IGNORECASE)
        # Fix C4 pe tokeni: ':' urmat de literă → 'Ș'
        t2 = re.sub(r':(?=[A-ZĂÂÎȚȘa-zăâîțș])', 'Ș', t2)
        # Elimină ghilimelele de la începtul tokenilor (ex: 'Popa → Popa)
        t2 = t2.lstrip("'\"\u2018\u2019\u201c\u201d")
        normalized_toks.append(t2)
    toks = normalized_toks

    full = " ".join(toks).upper()

    f = {k: "Nedetectat" for k in ("Nume", "Prenume", "CNP", "Serie", "Numar", "Sex", "Adresa")}

    # ── Etichete spatiale ──
    for i, tok in enumerate(toks):
        up = tok.upper()

        if re.search(r"\b[NWHMB]UM[EĂO/]", up) and not re.search(r"\bPR[EO]NUM", up) and f["Nume"] == "Nedetectat":
            # Cuvinte de pe eticheta cărții care pot fi confundate cu Numele
            _INVALID_NUME = {"IDENTIT", "IDENTITY", "IDENTITATE", "CARTE", "CARD", "BIRTH", "PLACE", "CNP",
                             "NAME", "NOM", "NOME", "LAST", "PRVI", "NATIONALITY", "CITIZEN",
                             "Y1S5H", "YISSH", "YIISH", "YISSH1", "YIISH1",
                             # Fix C1: etichete multi-limbă + prescurtări adresă
                             "ROUMANIE", "ROMANIA", "ROMIAIA", "ROMINIA", "RONGADIA",
                             "SEX", "SEXE", "SEXELSEX",
                             "IBIRTH", "IBÎRTH", "IBIRIH",
                             "PRENOM", "PRENUME", "FIRST", "FORENAME",
                             "PRENUMELPRENOM", "NUMELNOM",
                             "ORȘ", "ORS", "MUN", "JUD", "STR", "SAT", "COM", "SECT",
                             "PH", "BV", "CJ", "CT", "IS", "TM", "DJ", "GL",  # abrevieri județe
                             "ROMÂNĂ", "ROMANA", "ROU", "NATIONALITY",
                             "JDENTITY", "IDENTIIAIE", "IDENTITYI",
                             "D'IDENTITE", "D'DENTITE", "DIDENTITE",
                             "VALABILITATE", "VALIDITE", "VALIDITY"}
            # Subșiruri care indică o etichetă, nu un nume
            # "PRENU" prinde corupțiile OCR ale etichetei "Prenume": PRENUMO, PRENUMA, PRENUMELP etc.
            _INVALID_SUBSTRINGS = ("PRENU", "PRENOM", "BIRTH", "ROUMAN", "IDENT", "SEXE",
                                   "VALID", "NAISS", "NATIO", "DOMICIL")
            for j in range(i + 1, min(i + 7, len(toks))):
                up_tok = toks[j].upper().strip()
                # Extrage primul cuvânt din token (OCR poate concatena mai multe cuvinte)
                first_word = re.split(r'[\s/\-,]+', up_tok)[0]
                candidate = None
                if re.match(r"^[A-ZĂÂÎȘȚÎ\-]{2,}$", first_word) and first_word not in _INVALID_NUME:
                    candidate = first_word
                elif re.match(r"^[A-ZĂÂÎȘȚ\-]{2,}$", up_tok) and up_tok not in _INVALID_NUME:
                    candidate = up_tok
                # Respinge candidați care conțin subșiruri de etichetă
                if candidate and any(sub in candidate for sub in _INVALID_SUBSTRINGS):
                    candidate = None
                if candidate:
                    candidate = re.sub(r'([A-ZĂÂÎȘȚ])\1$', r'\1', candidate)
                    candidate = re.sub(r'IUI$', 'IU', candidate)
                    f["Nume"] = candidate; break

        if re.search(r"\bPR[EO]NUM", up) and f["Prenume"] == "Nedetectat":
            # Cuvinte care apar pe etichetă multi-limbă și pot fi citite greșit ca prenume
            _INVALID_PRENUME = {"NAME", "NAMO", "NOME", "NOMI", "PRENOM", "FIRST", "BIRTH", "PLACE",
                                "LAST", "NOM", "PREN", "RENOM", "FORENAME", "DATE", "ISSUED", "SEX"}
            for j in range(i + 1, min(i + 8, len(toks))):
                up_tok = toks[j].upper().strip()
                if up_tok in _INVALID_PRENUME:
                    continue  # sărim artefactele din etichetă
                # Tokenul poate conține spații (OCR combină cuvinte) sau litere mici
                # → luăm primul cuvânt care arată ca un prenume valid
                words = re.split(r'[\s,]+', up_tok)
                candidate = None
                for w in words:
                    if w in _INVALID_PRENUME:
                        continue
                    if re.match(r'^[A-ZĂÂÎȘȚ][A-ZĂÂÎȘȚ\-]{1,}$', w):
                        candidate = w; break
                if candidate:
                    val = candidate
                    val = re.sub(r'([A-ZĂÂÎȘȚ])\1$', r'\1', val)  # strip trailing duplicate
                    val = re.sub(r'IUI$', 'IU', val)               # IUI → IU (OCR artifact)
                    f["Prenume"] = val; break

        if re.search(r"[SBAb][EO][XK]", up) and f["Sex"] == "Nedetectat":
            # Caută M/F în tokenii următori (sorted by line)
            for j in range(i + 1, min(i + 10, len(toks))):
                v = toks[j].strip().upper()
                if v in ("M", "F"):
                    f["Sex"] = v; break
                if v in ("MASCULIN", "BARBATESC"):
                    f["Sex"] = "M"; break
                if v in ("FEMININ", "FEMEIESC"):
                    f["Sex"] = "F"; break
            # Fallback spațial: caută M/F printre toți tokenii scurți din imagine
            if f["Sex"] == "Nedetectat":
                for r in results:
                    v = r[1].strip().upper()
                    if v in ("M", "F"):
                        f["Sex"] = v; break

    # ── Regex pe textul complet ──

    # CNP: 13 cifre, prima 1-9
    m = re.search(r"\b([1-9]\d{12})\b", full)
    if m: f["CNP"] = m.group(1)

    # ── Serie + Număr: prioritate MRZ (linia 2), fallback etichetă ──
    # MRZ linia 2 începe cu seria (2 litere) + numărul (6 cifre) + '<' + ...
    # Exemplu: PX968762<4ROU...
    # Este cea mai fiabilă sursă, neafectată de reordonarea ymid.
    _INVALID_SERIE = {"NA", "NR", "DE", "LA", "LE", "UN", "NU", "SE", "OR", "EU",
                      "ID", "RO", "AD", "AP", "NO"}

    # 1) MRZ: 2 litere urmate imediat de 6 cifre (fără separator)
    mrz_sn = re.search(r"\b([A-Z]{2})(\d{6})<", full)
    if mrz_sn and mrz_sn.group(1) not in _INVALID_SERIE:
        f["Serie"] = mrz_sn.group(1)
        f["Numar"] = mrz_sn.group(2)

    # 2) Fallback etichetă: SERIA/SERIES + 2 litere (tolerăm și SERLA/SERJA = OCR corruption)
    if f["Serie"] == "Nedetectat":
        m = re.search(r"SER[ILJ][AE][:\s/]*(?:\S+\s+){0,3}?([A-Z]{2})\b", full)
        if m and m.group(1) not in _INVALID_SERIE:
            f["Serie"] = m.group(1)

    # 3) Fallback număr: NR + 6 cifre sau 6 cifre după serie
    if f["Numar"] == "Nedetectat":
        m = re.search(r"NR\.?\s*(\d{6})", full)
        if m:
            f["Numar"] = m.group(1)
        elif f["Serie"] != "Nedetectat":
            m = re.search(re.escape(f["Serie"]) + r"(\d{6})", full)
            if m: f["Numar"] = m.group(1)
    # 4) Fallback: orice 6 cifre standalone în zona header-ului (primii 40% tokeni)
    if f["Numar"] == "Nedetectat":
        header_toks = toks[:max(1, len(toks) * 2 // 5)]
        for tok in header_toks:
            m = re.fullmatch(r"(\d{6})", tok.strip())
            if m:
                f["Numar"] = m.group(1)
                break

    # Adresă: text după DOMICILIU (+ orice sufix corupt + etichete multilingve concatenate)
    # până la EMIS/VALABIL/sfârşit.
    # Exemplu corupt: "DOMICILLU/ADROSSELADDROSS JUD PH..." → consumăm tot prefixul etichetei.
    m = re.search(
        r"(?:DOMICIL\w{0,6}(?:[/\s]*(?:L?ADRESS\w*|ADDRESS\w*|ADROSSE\w*))*"
        r"|DORNOUHU\w*|DORNOITN\w*|DOMICL\w*"
        r"|L?ADRESS[EẼ]?S?|ADDRESS)"
        r"\s*[:/]?\s*(.*?)(?=\s*(?:EMIS[AĂ]?|EMLSA|EOUSS|VALABIL|ISSUED|VALID|SPCLEP|POLI[TȚ]IA|SPC|EUP|EUV|EVO)|\Z)",
        full, re.DOTALL | re.IGNORECASE
    )
    if m:
        adr = re.sub(r"\s+", " ", m.group(1)).strip()
        adr = adr.lstrip("./: ")
        # Dezlipire token OCR comprimat: sc.Aap.10 → sc.A ap.10, scAap10 → sc.A ap.10
        # Cazul 1: prefix(sc/bl/et) + punct opțional + literă + prefix2(ap/sc/bl/et) + punct + cifre
        adr = re.sub(r'\b(sc|bl|et)\.?([A-Z])(ap|bl|sc|et)\.?(\d)',
                     r'\1.\2 \3.\4', adr, flags=re.IGNORECASE)
        # Cazul 2: ap/sc/bl/et urmat de literă lipită direct de altceva
        adr = re.sub(r'\b(ap|sc|bl|et)\.?([A-Z])(?=\d)',
                     r'\1.\2 ', adr, flags=re.IGNORECASE)
        prev = None
        while prev != adr:
            prev = adr
            adr = re.sub(
                r"^[/:\s]*(?:L\s*)?(?:DOMICIL\w{0,6}|DORNOUHU\w*|DORNOITN\w*|ADRESS?[EO]?\w*|ADD?R[EO]?SS?\w*|ADDRESS\w*|DRESS\w+|RESSO\w*)\s*[/:\s]*",
                "", adr, flags=re.IGNORECASE
            ).strip().lstrip("./: ")
        # Corecție OCR frecventă: $ → Ș, : urmat de literă → Ș (Fix C4)
        adr = adr.replace('$', 'Ș')
        adr = re.sub(r':(?=[A-ZĂÂÎȚȘ])', 'Ș', adr)
        # Eliminare reziduuri de la final
        adr = re.sub(r'(?:EMIS[AĂ]?|EMLSA|EOUSS|VALABIL|ISSUED|VALID|SPCLEP|POLI[TȚ]IA|SPC|EUP|EUV|EVO).*$', '', adr, flags=re.IGNORECASE).strip()
        # Trunchierea inteligentă GREEDY: păstrăm până la ULTIMUL ap/bl/sc/et + număr sau literă
        m_last = re.search(r'^(.*\b(?:AP|BL|SC|ET)\.?\s*[A-Z]?\d+\w*'
                           r'|.*\b(?:AP|BL|SC|ET)\.?\s*[A-Z]\b)',
                           adr, re.IGNORECASE | re.DOTALL)
        if m_last:
            adr = m_last.group(1).strip()
            with open("debug_adr.txt","a",encoding="utf-8") as _df: _df.write(f"[D2] POST-GREEDY: {adr!r}\n")
        else:
            m_nr = re.search(r'^(.*\bNR\.?\s*\d+\w*)', adr, re.IGNORECASE | re.DOTALL)
            if m_nr:
                adr = m_nr.group(1).strip()
                with open("debug_adr.txt","a",encoding="utf-8") as _df: _df.write(f"[D2] POST-NR-TRUNC: {adr!r}\n")
            else:
                with open("debug_adr.txt","a",encoding="utf-8") as _df: _df.write(f"[D2] NO-TRUNC: {adr!r}\n")
        # Eliminare junk OCR rămas la final
        adr = re.sub(r'\s+\S{15,}\s*$', '', adr).strip()
        adr = re.sub(r'\s+[A-Z0-9]{10,}\s*$', '', adr).strip()
        adr = re.sub(r'[<{]{2,}.*$', '', adr).strip()
        adr = re.sub(r'[,\s]+\d{3,6}[,\s]+[A-Z0-9]{2,5}\s*$', '', adr).strip()
        adr = re.sub(r'[,\s]+[A-Z]{2,3}\s*$',               '', adr).strip()
        adr = re.sub(r'[,\s]+\d{3,6}\s*$',                  '', adr).strip()
        with open("debug_adr.txt","a",encoding="utf-8") as _df: _df.write(f"[D3] POST-CLEANUP: {adr!r}\n")
        # Fix: elimină cod poștal + abreviere județ interpolate între localitate și stradă
        # ex. "ORȘ BĂICOI 582 PH STR..." → "ORȘ BĂICOI STR..."
        _STR_MARKERS = r'(?:STR|BD|BDUL|BULEVARD|CALEA|ALEEA|INTRAREA|SPLAIUL|DRUMUL|PIAȚA|SOSEAUA)\b'
        adr = re.sub(
            r'\s+\d{3,6}\s+[A-Z]{2,3}\s+(?=' + _STR_MARKERS + r')',
            ' ', adr, flags=re.IGNORECASE
        )
        # Același pattern dar cu cifre singure la final (ex. "582" fără cod județ înaintea STR)
        adr = re.sub(
            r'\s+\d{3,6}\s+(?=' + _STR_MARKERS + r')',
            ' ', adr, flags=re.IGNORECASE
        )

        # Validare calitate: adresa trebuie să conțină cel puțin un marcator de adresă
        # Dacă nu începe cu marcatorul, curățăm prefixele scurte reziduale (junk OCR)
        if len(adr) > 5:
            # Curățare suplimentară: prefixe reziduale scurte sau cuvinte-junk cunoscute
            # inclusiv reziduuri tip 'ADDRESE', 'ADRESSE' care nu au fost prinse de while-loop
            for _ in range(6):
                m_pfx = re.match(
                    r'^(?:[A-Z]{1,3}\s+'
                    r'|(?:OL|VA|LE|LA|UN|DE|DU|DI|EL|IL|AN|NU|AL|LI|VE|OE|AE)\s+'
                    r'|ADD?R[EO]?SS?[EO]?\w*\s*[/:\s]*'
                    r'|ADRESS?[EO]?\w*\s*[/:\s]*)',
                    adr, re.IGNORECASE
                )
                if m_pfx:
                    adr = adr[m_pfx.end():].lstrip('./: ')
                else:
                    break
            # Acceptăm dacă adresa începe SAU conține un marcator valid de adresă
            _ADR_MARKER = re.compile(
                r'\b(?:JUD|MUN|STR|BD|SAT|OR[SȘ]|SECT|SOS|CALEA|ALEEA|INTRAREA|SPLAIUL|DRUMUL)\b',
                re.IGNORECASE
            )
            if _ADR_MARKER.search(adr):
                f["Adresa"] = adr

    # ── Fallback adresă: construiește adresa token-by-token când DOMICILIU e corupt ──
    if f["Adresa"] == "Nedetectat":
        _ADR_LOC = re.compile(
            r'\bJUD[.\s]{0,2}[A-ZĂÂÎȘȚ]{1,4}'
            r'[.\s]{0,3}(?:MUN|HUN|SAT|OR[SȘ]|COM|SECT?|MUNICIPIUL?|ORA[ȘS]UL?)',
            re.I
        )
        # Fix C7: pattern care acceptă JUD și loc type în tokeni SEPARATI
        _ADR_JUD_ONLY = re.compile(r'\bJUD\.?\s*([A-ZĂÂÎȘȚ]{2,3})\b', re.I)
        _ADR_LOC_TYPE = re.compile(
            r'^\s*(?:MUN|HUN|SAT|OR[SȘ]|COM|SECT?|MUNICIPIUL?|ORA[ȘS]UL?)\.?\s*', re.I
        )
        # SUR = corupție OCR frecventă a lui STR pe imagini blur
        _ADR_STR = re.compile(
            r'\b(?:STR|SUR|BD|BDUL|CALEA|INTRAREA|ALEEA|SPLAIUL|DRUMUL)\b', re.I
        )
        # Colectăm TOȚI tokenii JUD valizi, alegem cel mai din josul paginii (domiciliu, nu naștere)
        loc_candidates = [(i, tok) for i, tok in enumerate(toks) if _ADR_LOC.search(tok)]

        # Fix C7: dacă nu am găsit JUD+LOC_TYPE în același token, căutăm în tokeni consecutivi
        if not loc_candidates:
            for i, tok in enumerate(toks):
                m_jud = _ADR_JUD_ONLY.search(tok)
                if m_jud:
                    # Cautăm loc type în următorii 3 tokeni
                    for j2 in range(i + 1, min(i + 4, len(toks))):
                        if _ADR_LOC_TYPE.match(toks[j2]):
                            # Combinăm: "Jud PH" + "Orș Băicoi" → "Jud PH Orș Băicoi"
                            combined = tok + " " + toks[j2]
                            # Tokenii de după loc type vor fi localitate
                            for k2 in range(j2 + 1, min(j2 + 3, len(toks))):
                                next_t = toks[k2].strip()
                                if _ADR_STR.search(next_t) or _ADR_JUD_ONLY.search(next_t):
                                    break
                                if re.match(r'^[A-ZĂÂÎȘȚa-zăâîșț\-\.\s]{2,}$', next_t):
                                    combined += " " + next_t
                                else:
                                    break
                            loc_candidates.append((i, combined))
                            break

        # Preferăm ultimul token JUD găsit (cel mai probabil e domiciliul, nu locul nașterii)
        for loc_i, loc_tok in reversed(loc_candidates):
            str_idx = None
            # Căutăm STR în tokenul combinat sau în tokenii următori
            search_start = loc_i + 1 if loc_tok == toks[loc_i] else loc_i
            for j in range(search_start, min(loc_i + 10, len(toks))):
                if _ADR_STR.search(toks[j]):
                    str_idx = j; break
            if str_idx is None:
                continue
            # Construim adresa: JUD token + direct STR token (fără tokeni intermediari care pot fi gunoi)
            parts = [loc_tok, toks[str_idx]]
            for k in range(str_idx + 1, min(str_idx + 4, len(toks))):
                nxt = toks[k].strip()
                if re.match(r'^[\w\d\s\.\-\/,]{1,30}$', nxt) and not re.search(r'[a-z]{5,}', nxt):
                    parts.append(nxt)
                else:
                    break
            adr = re.sub(r"\s+", " ", " ".join(parts)).strip().upper()
            adr = adr.replace('$', 'Ș')
            adr = re.sub(r'\bSUR\b', 'STR', adr)
            adr = re.sub(r'\bHUN\b', 'MUN', adr)
            adr = re.sub(r'[,\s]+\d{3,6}\s*$', '', adr).strip()
            adr = re.sub(r'[,\s]+[A-Z]{2,3}\s*$', '', adr).strip()
            if len(adr) > 10:
                f["Adresa"] = adr
                break


    # ── Fallback MRZ pentru Nume, Prenume și SN ──
    # MRZ linia 1: ID+ROU+NUME+<<+PRENUME
    # MRZ linia 2: SB945253<7ROU...
    # Strategie: Luăm pozele din 30% de jos și le grupăm pe linii
    res_sorted_all = sorted(results, key=lambda r: (r[0][0][1] + r[0][2][1])/2)
    h_max = max([max(r[0], key=lambda p: p[1])[1] for r in results]) if results else 1000
    
    bottom_toks = _sort_tokens_by_line([r for r in results if (r[0][0][1] + r[0][2][1])/2 > h_max * 0.7], overlap_thresh=0.6)
    full_mrz_bottom = "".join(bottom_toks).upper().replace(' ', '')
    
    # regex mrz1 (Nume/Prenume)
    mrz1 = re.search(r"[^A-Z]?ID[A-Z]{3}([A-Z]{2,}?)<<([A-Z<]+?)<{2,}", full_mrz_bottom)
    if not mrz1:
        # Fallback pe tot textul dacă scanarea a eșuat local
        mrz_all_clean = "".join([r[1].upper().replace(' ', '') for r in results])
        mrz1 = re.search(r"[^A-Z]?ID[A-Z]{3}([A-Z]{2,}?)<<([A-Z<]+?)<{2,}", mrz_all_clean)

    if mrz1:
        if f["Nume"] == "Nedetectat":
            f["Nume"] = mrz1.group(1)
        # MRZ suprascrie Prenume dacă: (a) nu e detectat, sau (b) MRZ dă un rezultat
        # mai lung (prenume compus: "ANDREI-LAURENTIU" > "ANDREI" extras din etichetă)
        prenume_mrz = mrz1.group(2).replace("<", "-").strip("-")
        if len(prenume_mrz) >= 2:
            if f["Prenume"] == "Nedetectat" or len(prenume_mrz) > len(f["Prenume"]):
                f["Prenume"] = prenume_mrz

    # ── Fallback MRZ pentru Sex și Serie/Număr (Line 2) ──
    # Exemplu: SB945253<7ROU0506166M300616252958924
    mrz2 = re.search(r"([A-Z]{2})(\d{6})<\d[A-Z0-9]{3}\d{7}([MF])\d{6}", full_mrz_bottom)
    if not mrz2:
         mrz_all_clean = "".join([r[1].upper().replace(' ', '') for r in results])
         mrz2 = re.search(r"([A-Z]{2})(\d{6})<\d[A-Z0-9]{3}\d{7}([MF])\d{6}", mrz_all_clean)
    
    if mrz2:
        if f["Serie"] == "Nedetectat": f["Serie"] = mrz2.group(1)
        if f["Numar"] == "Nedetectat": f["Numar"] = mrz2.group(2)
        if f["Sex"]   == "Nedetectat": f["Sex"]   = mrz2.group(3)

    # ── Restaurare diacritice: OCR-first + reguli lingvistice fallback ──
    #
    # Pasul 1: construim index ASCII→original din tokenii EasyOCR bruti
    # Pasul 2: căutăm în index (EasyOCR poate citi diacriticele direct când imaginea e clară)
    # Pasul 3: dacă OCR tot returnează ASCII, aplicăm regulile lingvistice românești

    _ocr_index: dict[str, str] = {}
    for r in results:
        raw = r[1].strip()
        if not raw:
            continue
        for word in re.split(r'[\s/\-,.()\[\]]+', raw):
            word = word.strip()
            if len(word) >= 2:
                key = _strip_accents(word.upper())
                existing = _ocr_index.get(key, "")
                # Preferăm versiunea cu mai multe caractere speciale românești
                score_new = sum(1 for c in word if c in 'ĂÂÎȘȚăâîșț')
                score_old = sum(1 for c in existing if c in 'ĂÂÎȘȚăâîșț')
                if score_new >= score_old:
                    _ocr_index[key] = word

    def _restore_word(ascii_word: str) -> str:
        """OCR lookup → reguli lingvistice → returnează forma cu diacritice."""
        key = _strip_accents(ascii_word.upper())
        ocr_w = _ocr_index.get(key, "")
        if ocr_w and any(c in 'ĂÂÎȘȚăâîșț' for c in ocr_w):
            return ocr_w.upper()
        # Fallback: reguli lingvistice
        return _apply_diacritic_rules(ascii_word)

    def _restore_name(name: str) -> str:
        """Aplică _restore_word pe fiecare componentă a unui nume compus."""
        return '-'.join(_restore_word(p) for p in name.split('-'))

    if f["Prenume"] != "Nedetectat":
        # Prenume: dicționar exact mai întâi, apoi OCR+reguli
        parts = f["Prenume"].split('-')
        out = []
        for part in parts:
            key = _strip_accents(part.upper())
            if key in _PRENUME_LOOKUP:
                out.append(_PRENUME_LOOKUP[key])
            else:
                out.append(_restore_word(part))
        f["Prenume"] = '-'.join(out)

    if f["Nume"] != "Nedetectat":
        f["Nume"] = _restore_name(f["Nume"])

    # ── Corecție nume de localități frecvent greșite de OCR ──
    _CITY_FIXES = {
        r'\bSIBIY\b':     'SIBIU',
        r'\bBRASO[YV]\b': 'BRASOV',
        r'\bDRASO[YV]\b': 'BRASOV',
        r'\bBRASO\b':     'BRASOV',
        r'\bCLUY\b':      'CLUJ',
        r'\bTIMI[SȘ]OARA\b': 'TIMISOARA',
        r'\bPLOIE[SȘ]TI\b':  'PLOIESTI',
        r'\bCONSTANTA\b':    'CONSTANTA',
        r'\bBUCURE[SȘ]TI\b': 'BUCURESTI',
        r'\bONE[SȘ]TI\b':    'ONESTI',
    }
    if f["Adresa"] != "Nedetectat":
        for pat, repl in _CITY_FIXES.items():
            f["Adresa"] = re.sub(pat, repl, f["Adresa"], flags=re.IGNORECASE)

    # ── Normalizare format JUD în adresă ──
    # OCR poate alătura "Jud" și abrevierea județului fără spațiu ("JUDPH" → "JUD. PH")
    # Fix P1: IGNORECASE + pontși formă compactă Jud.BC fără spațiu
    if f["Adresa"] != "Nedetectat":
        f["Adresa"] = re.sub(
            r'\bJUD\.?\s*([A-Z]{2,3})\b', r'JUD. \1',
            f["Adresa"], flags=re.IGNORECASE
        ).upper()
        # Fix C2: eliminăm emitentul SPCLEP/POLITIA/EMISA din adresa finală
        f["Adresa"] = re.sub(
            r'\s*(?:SPCLEP|POLI[TȚ]IA|EMIS[AĂ]?)\b.*$', '',
            f["Adresa"], flags=re.IGNORECASE
        ).strip()
        # Fix P2: elimină orice apare după numărul casei (NR.X) dacă nu e BL/SC/AP/ET
        # ex. "NR.18 582" → "NR.18", "NR 34 589 EV2" → "NR 34"
        # Notă: OCR concatenează frecvent BL cu nr/scara (BLST6, BL3, BL.C) → nu cerem \b
        def _strip_after_nr(addr: str) -> str:
            m_nr = re.search(r'\bNR\.?\s*\d+\w*', addr, re.IGNORECASE)
            if not m_nr:
                return addr
            after = addr[m_nr.end():].strip()
            if after and not re.match(r'(?:BL|SC|AP|ET)', after, re.IGNORECASE):
                return addr[:m_nr.end()].strip()
            return addr
        f["Adresa"] = _strip_after_nr(f["Adresa"])

    # ── Segmentare adresă ──
    if f["Adresa"] != "Nedetectat":
        seg = parse_address(f["Adresa"])
        f["Judet"]  = seg["Judet"]
        f["Strada"] = seg["Strada"]
        # Restaurare diacritice Localitate: OCR-first, fallback reguli lingvistice
        loc_raw = seg["Localitate"]
        if loc_raw and loc_raw != "Nedetectat":
            loc_words = loc_raw.split()
            loc_out = []
            for w in loc_words:
                key = _strip_accents(w.upper())
                ocr_w = _ocr_index.get(key, "")
                if ocr_w and any(c in 'ĂÂÎȘȚăâîșț' for c in ocr_w):
                    loc_out.append(ocr_w.title())
                else:
                    # Aplică regulile de localitate pe cuvântul individual
                    restored = _restore_localitate_rules(w.upper())
                    loc_out.append(restored)
            f["Localitate"] = " ".join(loc_out)

            # Fix C8: corecții OCR frecvente de localități
            _LOC_OCR_FIXES = {
                "Băjcoi": "Băicoi", "Bajcoi": "Băicoi", "Baicoi": "Băicoi",
                "Campina": "Câmpina", "Câmp-": "Câmpina",
                "Ploesti": "Ploiești", "Pitesti": "Pitești",
                "Bucuresti": "București", "Brasov": "Brașov",
                "Timisoara": "Timișoara", "Constanta": "Constanța",
            }
            loc_val = f["Localitate"]
            for wrong, correct in _LOC_OCR_FIXES.items():
                loc_val = re.sub(re.escape(wrong), correct, loc_val, flags=re.IGNORECASE)
            f["Localitate"] = loc_val
        else:
            f["Localitate"] = loc_raw
    else:
        f["Judet"]      = "Nedetectat"
        f["Localitate"] = "Nedetectat"
        f["Strada"]     = "Nedetectat"

    return f


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_EASYOCR_READER = None
_EASYOCR_INIT_LOCK = threading.Lock()
_EASYOCR_PROCESS_LOCK = threading.Lock()


def get_easyocr_reader():
    """Inițializează o singură dată reader-ul EasyOCR și îl refolosește."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        with _EASYOCR_INIT_LOCK:
            if _EASYOCR_READER is None:
                try:
                    import easyocr
                    import torch
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "Lipseste dependinta pentru OCR. Instaleaza pachetul 'easyocr' in acelasi mediu Python din care ruleaza API-ul."
                    ) from exc
                gpu = torch.cuda.is_available()
                _EASYOCR_READER = easyocr.Reader(["ro"], gpu=gpu)
    return _EASYOCR_READER


def _process_single(img_path: str, reader, debug_prefix: str = "") -> dict | None:
    """
    Procesează o singură imagine și returnează dict-ul cu rezultate.
    Dacă debug_prefix e gol, nu salvează fișierele debug_card/debug_proc.
    """
    buf: list[str] = []           # buffer – tot output-ul se afișează atomic la final

    buf.append(f"\n{'='*55}\n  Procesăm: {os.path.basename(img_path)}\n{'='*55}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[EROARE] Nu pot citi imaginea: {img_path}")
        return None

    # ── Detectare & corecție unghi ──
    t_deskew = time.perf_counter()
    card, method = deskew_and_crop(img_bgr)
    t_deskew = round(time.perf_counter() - t_deskew, 3)
    buf.append(f"[INFO] Detecție unghi: {method}  ({t_deskew}s)")
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_card.jpg", card)

    # ── Preprocesare ──
    t_prep = time.perf_counter()
    ocr_img = preprocess(card)
    t_prep = round(time.perf_counter() - t_prep, 3)
    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_proc.jpg", ocr_img)

    # ── OCR cu suprimarea output-ului GPU la nivel OS (fd-level) ──
    # EasyOCR/PyTorch scriu prin C++ direct pe fd=1/2, bypass-ând sys.stdout.
    # Singura metodă fiabilă: redirect fd 1+2 → devnull în jurul readtext().
    t_ocr = time.perf_counter()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _saved_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(_devnull, 1); os.dup2(_devnull, 2)
        results = reader.readtext(ocr_img, detail=1)
        
        # Fallback P12: Dacă OCR eșuează complet (0-4 tokeni valizi), probabil e rotit 180
        if not results or len(results) < 5:
            ocr_flipped = cv2.rotate(ocr_img, cv2.ROTATE_180)
            res_flipped = reader.readtext(ocr_flipped, detail=1)
            # Dacă rotirea a adus mult mai mulți tokeni, o acceptăm
            if len(res_flipped) > (len(results) or 0) + 10:
                results = res_flipped
                method += "+rot180_ocr_fallback"
                # Actualizăm și ocr_img pentru log-ul debug_proc (opțional, dar util pt debug)
                ocr_img = ocr_flipped
    finally:
        os.dup2(_saved_fds[0], 1); os.dup2(_saved_fds[1], 2)
        os.close(_devnull)
        os.close(_saved_fds[0]); os.close(_saved_fds[1])
    t_ocr = round(time.perf_counter() - t_ocr, 3)

    buf.append(f"[INFO] Imagine OCR: {ocr_img.shape[1]}×{ocr_img.shape[0]} px  ({t_prep}s)")

    toks = [r[1] for r in results]
    buf.append(f"[DEBUG] Tokeni detectați ({len(toks)}, {t_ocr}s):")
    for i, t in enumerate(toks):
        buf.append(f"  [{i:02d}] {t}")

    # ── Early-exit imagine ilizibilă (Fix P3) ──
    valid_toks = [t for t in toks if re.search(r'[A-Za-zĂÂÎȘȚăâîşţ]{3,}', t)]
    # ── Fallback post-OCR: warp greșit → reîncercăm cu originalul ──
    # Dacă metoda a folosit warp(color) sau warp AND avem puțini tokeni valizi,
    # înseamnă că warp-ul a fost o detecție falsă. Reîncercăm pe imaginea originală.
    if len(valid_toks) < 8 and "warp" in method.lower():
        buf.append(f"[WARN] Warp slab ({len(valid_toks)} tokeni valizi) — reîncercăm cu rotație.")
        # deskew_and_crop cu skip_warp=True: sare warp-urile (Strategy 0,1) și
        # aplică direct rotația Hough (Strategy 2) — ideal pt carduri la unghi moderat.
        card_rot, method_rot = deskew_and_crop(img_bgr, skip_warp=True)
        ocr_orig = preprocess(card_rot)
        _devnull2 = os.open(os.devnull, os.O_WRONLY)
        _saved_fds2 = [os.dup(1), os.dup(2)]
        try:
            os.dup2(_devnull2, 1); os.dup2(_devnull2, 2)
            res_orig = reader.readtext(ocr_orig, detail=1)
        finally:
            os.dup2(_saved_fds2[0], 1); os.dup2(_saved_fds2[1], 2)
            os.close(_devnull2)
            os.close(_saved_fds2[0]); os.close(_saved_fds2[1])
        valid_orig = [t for r in res_orig for t in [r[1]] if re.search(r'[A-Za-zĂÂÎȘȚăâîşţ]{3,}', t)]
        if len(valid_orig) > len(valid_toks):
            buf.append(f"[INFO] Rotație mai bună ({len(valid_orig)} tokeni, {method_rot}) — folosim rotația.")
            results    = res_orig
            ocr_img    = ocr_orig
            toks       = [r[1] for r in results]
            method     = f"rot_fallback({method_rot})"
            valid_toks = valid_orig

    if len(valid_toks) < 5:
        buf.append(f"[WARN] Imagine ilizibilă ({len(valid_toks)} tokeni valizi) — sărită.")
        import sys; sys.stdout.write("\n".join(buf) + "\n"); sys.stdout.flush()
        return None

    # ── Extragere câmpuri ──
    t_extract = time.perf_counter()
    fields = extract_fields(results)
    t_extract = round(time.perf_counter() - t_extract, 3)

    t_total = round(t_deskew + t_prep + t_ocr + t_extract, 3)

    cnp_val  = fields["CNP"]
    cod_jud  = cnp_cod_judet(cnp_val)          # ex. "29"

    # Construim rezultatul cu Cod_judet înaintea lui Judet
    result = {
        "Fisier":     os.path.basename(img_path),
        "Nume":       fields["Nume"],
        "Prenume":    fields["Prenume"],
        "CNP":        cnp_val,
        "Serie":      fields["Serie"],
        "Numar":      fields["Numar"],
        "Sex":        fields["Sex"],
        "Adresa":     fields["Adresa"],
        "Cod_judet":  cod_jud,
        "Judet":      fields.get("Judet", "Nedetectat"),
        "Localitate": fields.get("Localitate", "Nedetectat"),
        "Strada":     fields.get("Strada", "Nedetectat"),
        "Metoda":     method,
        "t_total":    t_total,
    }

    buf.append("\n--- REZULTATE ---")
    for k, v in result.items():
        buf.append(f"  {k}: {v}")
    buf.append(f"\n  TOTAL: {t_total:.3f} s")

    # Scriem TOT output-ul atomic (un singur write) → evită intercalarea cu EasyOCR threads
    import sys
    sys.stdout.write("\n".join(buf) + "\n")
    sys.stdout.flush()

    return result


def extract_id_card_data(img_path: str, debug_prefix: str = "") -> dict | None:
    """
    Helper reutilizabil pentru API: procesează un singur buletin românesc.
    Lock-ul serializează inferența peste reader-ul partajat.
    """
    reader = get_easyocr_reader()
    with _EASYOCR_PROCESS_LOCK:
        return _process_single(img_path, reader, debug_prefix=debug_prefix)


def _save_excel(results: list[dict]):
    """Salvează lista de rezultate în Excel (append sau nou)."""
    df = pd.DataFrame(results)
    try:
        if not os.path.isfile(EXCEL_FILE):
            df.to_excel(EXCEL_FILE, index=False)
        else:
            with pd.ExcelWriter(EXCEL_FILE, mode="a", engine="openpyxl",
                                if_sheet_exists="overlay") as wr:
                sh = wr.book["Sheet1"] if "Sheet1" in wr.book.sheetnames else None
                sr = sh.max_row if sh else 0
                df.to_excel(wr, index=False, header=(sr == 0), startrow=sr)
        print(f"[INFO] Salvat în '{EXCEL_FILE}' ({len(results)} înregistrări)")
    except Exception as e:
        print(f"[WARN] Eroare Excel: {e}")


def _save_json(results: list[dict]):
    """Adaugă rezultatele în fișierul JSON (append la lista existentă)."""
    try:
        existing = []
        if os.path.isfile(JSON_FILE):
            with open(JSON_FILE, "r", encoding="utf-8") as jf:
                existing = json.load(jf)
                if not isinstance(existing, list):
                    existing = [existing]
        existing.extend(results)
        with open(JSON_FILE, "w", encoding="utf-8") as jf:
            json.dump(existing, jf, ensure_ascii=False, indent=2)
        print(f"[INFO] Salvat în '{JSON_FILE}'")
    except Exception as e:
        print(f"[WARN] Eroare JSON: {e}")


def main():
    default = os.path.join(ASSETS_DIR, "buletin2.jpg")
    user_in = input(
        f"Introdu calea unui fișier sau a unui folder\n"
        f"[Enter = '{default}']: "
    ).strip()
    path = user_in if user_in else default

    reader = get_easyocr_reader()

    # ── Mod BATCH: folder ──
    if os.path.isdir(path):
        images = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])
        if not images:
            print(f"[EROARE] Nu am găsit imagini în folderul: {path}")
            return

        print(f"\n[BATCH] {len(images)} imagini găsite în '{path}'")
        all_results = []
        for idx, img_path in enumerate(images, 1):
            print(f"\n[{idx}/{len(images)}] {os.path.basename(img_path)}")
            res = _process_single(img_path, reader, debug_prefix="")
            if res:
                all_results.append(res)

        if not all_results:
            print("[EROARE] Nicio imagine nu a putut fi procesată.")
            return

        print(f"\n{'='*55}")
        print(f"  BATCH COMPLET: {len(all_results)}/{len(images)} imagini procesate")
        print(f"{'='*55}")

        _save_json(all_results)

        save_xlsx = input("\nSalvezi și în Excel (.xlsx)? [y/N]: ").strip().lower()
        if save_xlsx in ("y", "yes", "da"):
            _save_excel(all_results)

    # ── Mod SINGLE: fișier ──
    elif os.path.isfile(path):
        res = _process_single(path, reader, debug_prefix="debug")
        if res:
            _save_json([res])
            save_xlsx = input("\nSalvezi și în Excel (.xlsx)? [y/N]: ").strip().lower()
            if save_xlsx in ("y", "yes", "da"):
                _save_excel([res])

    else:
        print(f"[EROARE] Calea nu există: {path}")


if __name__ == "__main__":
    main()
