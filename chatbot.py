import os
import json
import numpy as np
import google.generativeai as genai
import docx
import re
import textwrap
import pdfplumber
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()


class KanunRAGChatbot:
    def __init__(self, kanun_path="docs/kanun/Yasa1.docx", karar_folder="docs/yargitay"):
        """
        Genel kanunlar ve mahkeme kararları için RAG Chatbot'u başlatır.
        """
        self.gemini_key = os.getenv('GEMINI_KEY')
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_KEY in your .env file")

        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.embedding_model = "models/embedding-001"

        self.kanun_path = kanun_path
        self.karar_folder = karar_folder
        self.kanun_data = self._extract_kanun_data()
        self.kanun_data["kararlar"] = self._extract_kararlar()
        self.embeddings_df = self._generate_embeddings()

        # Chunking system variables
        self.history = []         # Henüz özetlenmemiş mesajlar (son chunk)
        self.summaries = []       # Chunk özetleri
        self.chunk_size = 10      # Her 10 mesajda bir özetle
        self.pinned_context = ""  # Ana konu (isteğe bağlı)

    def _extract_kanun_data(self):
        """Docx dosyasından kanun bilgilerini ve maddeleri çıkarır."""
        doc = docx.Document(self.kanun_path)
        kanun_data = {
            "kanun_adi": "",
            "kanun_numarasi": "",
            "yayim_tarihi": "",
            "resmi_gazete": "",
            "maddeler": [],
            "gecici_maddeler": [],
            "ekler": [],
            "kararlar": []
        }

        current_madde = None
        current_content = []
        section = "meta"  # meta, maddeler, gecici_maddeler, ekler

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Kanun meta verilerini çıkar
            if section == "meta":
                if "Kanun Numarası:" in text:
                    kanun_data["kanun_numarasi"] = text.split(":")[1].strip()
                elif text.startswith("**") and text.endswith("**"):
                    kanun_data["kanun_adi"] = text.strip("**").strip()
                elif "Yayımlandığı Resmî Gazete:" in text:
                    kanun_data["resmi_gazete"] = text.split(":", 1)[1].strip()
                elif "Yayımlandığı Düstur:" in text:
                    kanun_data["yayim_tarihi"] = text.split(":", 1)[1].strip()

            # Maddeleri tespit et
            if text.startswith("MADDE ") and "(Ek:" not in text:
                if current_madde:
                    if section == "maddeler":
                        kanun_data["maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                            "title": f"Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                    elif section == "gecici_maddeler":
                        kanun_data["gecici_maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                            "title": f"Geçici Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                match = re.match(r"MADDE (\d+)", text)
                if match:
                    current_madde = match.group(1)
                    current_content = [text.replace(f"MADDE {current_madde}", "").strip()]
                    section = "maddeler"

            # Geçici maddeleri tespit et
            elif text.startswith("GEÇİCİ MADDE "):
                if current_madde:
                    kanun_data["maddeler"].append({
                        "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                        "title": f"Madde {current_madde}",
                        "content": " ".join(current_content).strip(),
                        "keywords": self._extract_keywords(" ".join(current_content))
                    })
                match = re.match(r"GEÇİCİ MADDE (\d+)", text)
                if match:
                    current_madde = match.group(1)
                    current_content = [text.replace(f"GEÇİCİ MADDE {current_madde}", "").strip()]
                    section = "gecici_maddeler"

            # Ekler veya diğer bölümler
            elif text.startswith("**") and section != "meta":
                if current_madde:
                    if section == "maddeler":
                        kanun_data["maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                            "title": f"Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                    elif section == "gecici_maddeler":
                        kanun_data["gecici_maddeler"].append({
                            "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                            "title": f"Geçici Madde {current_madde}",
                            "content": " ".join(current_content).strip(),
                            "keywords": self._extract_keywords(" ".join(current_content))
                        })
                current_madde = None
                current_content = []
                section = "ekler"
                kanun_data["ekler"].append({
                    "title": text.strip("**").strip(),
                    "content": ""
                })

            # İçeriği ekle
            elif current_madde:
                current_content.append(text)
            elif section == "ekler" and kanun_data["ekler"]:
                kanun_data["ekler"][-1]["content"] += " " + text

        # Son maddeyi kaydet
        if current_madde:
            if section == "maddeler":
                kanun_data["maddeler"].append({
                    "id": f"{kanun_data['kanun_numarasi']}-MADDE-{current_madde}",
                    "title": f"Madde {current_madde}",
                    "content": " ".join(current_content).strip(),
                    "keywords": self._extract_keywords(" ".join(current_content))
                })
            elif section == "gecici_maddeler":
                kanun_data["gecici_maddeler"].append({
                    "id": f"{kanun_data['kanun_numarasi']}-GECICI-MADDE-{current_madde}",
                    "title": f"Geçici Madde {current_madde}",
                    "content": " ".join(current_content).strip(),
                    "keywords": self._extract_keywords(" ".join(current_content))
                })

        return kanun_data

    def _extract_keywords(self, text):
        """Basit anahtar kelime çıkarma."""
        common_keywords = [
            "tüketici", "sözleşme", "madde", "kanun", "cayma", "hakkı",
            "sorumluluk", "ayıp", "kredi", "faiz", "satıcı", "sağlayıcı",
            "yargıtay", "danıştay", "karar", "emsal"
        ]
        words = text.lower().split()
        return [word for word in common_keywords if word in words]

    def _extract_kararlar(self):
        """Karar dosyalarından bilgileri çıkarır."""
        kararlar = []
        if not os.path.exists(self.karar_folder):
            print(f"Karar klasörü ({self.karar_folder}) bulunamadı.")
            return kararlar

        for file in os.listdir(self.karar_folder):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.karar_folder, file)
                text = self._extract_pdf_text(file_path)
                
                # Yargıtay kararı için detaylı regex desenleri
                yargitay_patterns = {
                    "esas_no": r"Esas No\s*:\s*(\d+/\d+)",
                    "karar_no": r"Karar No\s*:\s*(\d+/\d+)",
                    "tarih": r"Tarih\s*:\s*(\d{2}\.\d{2}\.\d{4})",
                    "dava_turu": r"(Hukuk|Ceza)\s*Genel\s*Kurulu",
                    "ilgili_madde": r"(?:6502|6098|818|818\.|818\.\d+)\s*sayılı\s*(?:kanun|yasa|Yasa|Kanun).*?MADDE\s*(\d+)",
                    "karar_ozeti": r"(?:KARAR\s*ÖZETİ|ÖZET|KARAR)\s*:?\s*(.*?)(?=\n\n|\Z)",
                    "gerekce": r"(?:GEREKÇE|GEREKÇESİ)\s*:?\s*(.*?)(?=\n\n|\Z)"
                }

                # Danıştay kararı için detaylı regex desenleri
                danistay_patterns = {
                    "esas_no": r"Esas No\s*:\s*(\d+/\d+)",
                    "karar_no": r"Karar No\s*:\s*(\d+/\d+)",
                    "tarih": r"Tarih\s*:\s*(\d{2}\.\d{2}\.\d{4})",
                    "dava_turu": r"(İdari|Vergi)\s*Dava\s*Daireleri\s*Kurulu",
                    "ilgili_madde": r"(?:6502|6098|818|818\.|818\.\d+)\s*sayılı\s*(?:kanun|yasa|Yasa|Kanun).*?MADDE\s*(\d+)",
                    "karar_ozeti": r"(?:KARAR\s*ÖZETİ|ÖZET|KARAR)\s*:?\s*(.*?)(?=\n\n|\Z)",
                    "gerekce": r"(?:GEREKÇE|GEREKÇESİ)\s*:?\s*(.*?)(?=\n\n|\Z)"
                }

                # Mahkeme türünü belirle
                is_yargitay = "YARGITAY" in text.upper() or "YARGITAY" in file.upper()
                patterns = yargitay_patterns if is_yargitay else danistay_patterns

                # Bilgileri çıkar
                karar_info = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        karar_info[key] = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()

                # Karar içeriğini oluştur
                content_parts = []
                if karar_info.get("karar_ozeti"):
                    content_parts.append(f"KARAR ÖZETİ: {karar_info['karar_ozeti']}")
                if karar_info.get("gerekce"):
                    content_parts.append(f"GEREKÇE: {karar_info['gerekce']}")

                # Karar nesnesini oluştur
                karar = {
                    "id": f"{'YARGITAY' if is_yargitay else 'DANISTAY'}-{karar_info.get('esas_no', '').replace('/', '-')}",
                    "court": "Yargıtay" if is_yargitay else "Danıştay",
                    "esas_no": karar_info.get("esas_no", ""),
                    "karar_no": karar_info.get("karar_no", ""),
                    "tarih": karar_info.get("tarih", ""),
                    "dava_turu": karar_info.get("dava_turu", ""),
                    "ilgili_madde": f"6502-MADDE-{karar_info.get('ilgili_madde', '')}" if karar_info.get("ilgili_madde") else "",
                    "content": "\n\n".join(content_parts) if content_parts else text[:2000],
                    "keywords": self._extract_keywords(text)
                }
                kararlar.append(karar)

        return kararlar

    def _extract_pdf_text(self, pdf_path):
        """PDF dosyasından metin çıkarır."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    # Sayfa metnini çıkar
                    page_text = page.extract_text() or ""
                    
                    # Metni temizle ve düzenle
                    page_text = re.sub(r'\s+', ' ', page_text)  # Fazla boşlukları temizle
                    page_text = re.sub(r'([.!?])\s+', r'\1\n', page_text)  # Cümle sonlarını düzenle
                    page_text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', page_text)  # Sayı formatlarını düzelt
                    
                    text += page_text + "\n\n"
                
                # Genel temizlik
                text = re.sub(r'\n{3,}', '\n\n', text)  # Fazla satır sonlarını temizle
                text = text.strip()
                
                return text
        except Exception as e:
            print(f"PDF işlenirken hata ({pdf_path}): {e}")
            return ""

    def _generate_embedding(self, text):
        """Metin için gömme vektörü üretir."""
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Gömme üretirken hata: {e}")
            return [0.0] * 768  # Varsayılan sıfır vektörü

    def _generate_embeddings(self):
        """Maddeler ve kararlar için gömme vektörleri üretir veya yükler."""
        embeddings_file = "kanun_embeddings.json"
        existing_data = None

        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                print("Mevcut kanun_embeddings.json yüklendi.")
            except Exception as e:
                print(f"JSON yüklenirken hata: {e}. Yeni gömmeler üretilecek.")

        if existing_data:
            existing_maddeler = {m["id"]: m for m in existing_data.get("maddeler", [])}
            existing_gecici = {m["id"]: m for m in existing_data.get("gecici_maddeler", [])}
            existing_ekler = {(e.get("title") or ""): e for e in existing_data.get("ekler", [])}
            existing_kararlar = {k["id"]: k for k in existing_data.get("kararlar", [])}

            for madde in self.kanun_data["maddeler"]:
                if madde["id"] in existing_maddeler and "embedding" in existing_maddeler[madde["id"]]:
                    madde["embedding"] = existing_maddeler[madde["id"]]["embedding"]
                else:
                    madde["embedding"] = self._generate_embedding(madde["content"])

            for madde in self.kanun_data["gecici_maddeler"]:
                if madde["id"] in existing_gecici and "embedding" in existing_gecici[madde["id"]]:
                    madde["embedding"] = existing_gecici[madde["id"]]["embedding"]
                else:
                    madde["embedding"] = self._generate_embedding(madde["content"])

            for ek in self.kanun_data["ekler"]:
                ek_title = ek.get("title", "")
                if ek_title in existing_ekler and "embedding" in existing_ekler[ek_title] and ek["content"]:
                    ek["embedding"] = existing_ekler[ek_title]["embedding"]
                else:
                    ek["embedding"] = self._generate_embedding(ek["content"]) if ek["content"] else [0.0] * 768

            for karar in self.kanun_data["kararlar"]:
                if karar["id"] in existing_kararlar and "embedding" in existing_kararlar[karar["id"]]:
                    karar["embedding"] = existing_kararlar[karar["id"]]["embedding"]
                else:
                    karar["embedding"] = self._generate_embedding(karar["content"])
        else:
            for madde in self.kanun_data["maddeler"]:
                madde["embedding"] = self._generate_embedding(madde["content"])
            for madde in self.kanun_data["gecici_maddeler"]:
                madde["embedding"] = self._generate_embedding(madde["content"])
            for ek in self.kanun_data["ekler"]:
                ek["embedding"] = self._generate_embedding(ek["content"]) if ek["content"] else [0.0] * 768
            for karar in self.kanun_data["kararlar"]:
                karar["embedding"] = self._generate_embedding(karar["content"])

        try:
            with open(embeddings_file, "w", encoding="utf-8") as f:
                json.dump(self.kanun_data, f, ensure_ascii=False, indent=2)
            print(f"{len(self.kanun_data['maddeler'])} madde, "
                  f"{len(self.kanun_data['gecici_maddeler'])} geçici madde, "
                  f"{len(self.kanun_data['ekler'])} ek ve "
                  f"{len(self.kanun_data['kararlar'])} karar işlendi.")
        except Exception as e:
            print(f"JSON kaydederken hata: {e}")

        return self.kanun_data

    def find_best_passage(self, query, top_k=4):
        """En uygun maddeleri ve kararları bulur."""
        # Önce sorguda dava numarası veya spesifik detaylar var mı kontrol et
        case_patterns = {
            "esas_no": r"Esas No[:\s]*(\d+/\d+)",
            "karar_no": r"Karar No[:\s]*(\d+/\d+)",
            "tarih": r"(\d{2}\.\d{2}\.\d{4})",
            "dava_turu": r"(Hukuk|Ceza|İdari|Vergi)\s*(?:Genel\s*Kurulu|Dava\s*Daireleri\s*Kurulu)?",
            "ozel_detay": r"(?:Ford|BMW|Mercedes|Audi|Volkswagen|Toyota|Honda|Hyundai|Kia|Renault|Peugeot|Citroen|Fiat|Opel|Dacia|Nissan|Mitsubishi|Suzuki|Mazda|Subaru|Lexus|Infiniti|Jaguar|Land Rover|Volvo|Porsche|Ferrari|Lamborghini|Maserati|Bentley|Rolls-Royce|Aston Martin|Bugatti|McLaren|Lotus|Alfa Romeo|Lancia|Seat|Skoda|MINI|Smart|Jeep|Chrysler|Dodge|Cadillac|Chevrolet|GMC|Buick|Lincoln|Tesla|Rivian|Lucid|Polestar|NIO|XPeng|Li Auto|BYD|Geely|Great Wall|Chery|JAC|SAIC|Dongfeng|FAW|BAIC|Changan|Brilliance|Haval|Lynk & Co|Wey|ORA|Aion|Arcfox|Baojun|Borgward|Changan|Chery|Denza|Dongfeng|FAW|GAC|Geely|Great Wall|Haval|Hongqi|JAC|Lynk & Co|NIO|ORA|Roewe|SAIC|Wey|XPeng|Zotye)\s+[A-Za-z0-9\s-]+"
        }

        # Sorgudan spesifik detayları çıkar
        query_details = {}
        for key, pattern in case_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                query_details[key] = match.group(1) if len(match.groups()) > 0 else match.group(0)

        # Eğer spesifik detaylar varsa, önce kararları filtrele
        if query_details:
            filtered_items = []
            for item in self.kanun_data["kararlar"]:
                match_score = 0
                # Esas no eşleşmesi
                if query_details.get("esas_no") and item.get("esas_no") == query_details["esas_no"]:
                    match_score += 3
                # Karar no eşleşmesi
                if query_details.get("karar_no") and item.get("karar_no") == query_details["karar_no"]:
                    match_score += 3
                # Tarih eşleşmesi
                if query_details.get("tarih") and item.get("tarih") == query_details["tarih"]:
                    match_score += 2
                # Dava türü eşleşmesi
                if query_details.get("dava_turu") and query_details["dava_turu"].lower() in item.get("dava_turu", "").lower():
                    match_score += 2
                # Özel detay eşleşmesi (araç modeli vb.)
                if query_details.get("ozel_detay") and query_details["ozel_detay"].lower() in item.get("content", "").lower():
                    match_score += 4

                if match_score > 0:
                    item["match_score"] = match_score
                    filtered_items.append(item)

            if filtered_items:
                # Eşleşme puanına göre sırala
                filtered_items.sort(key=lambda x: x.get("match_score", 0), reverse=True)
                return filtered_items[:top_k]

        # Spesifik eşleşme yoksa veya yeterli sonuç bulunamadıysa, embedding tabanlı arama yap
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        all_items = (
            self.kanun_data["maddeler"] +
            self.kanun_data["gecici_maddeler"] +
            self.kanun_data["kararlar"]
        )
        
        # Önce kararları kontrol et
        karar_items = [item for item in all_items if "court" in item]
        if karar_items:
            karar_embeddings = np.array([m.get("embedding", [0.0] * 768) for m in karar_items])
            karar_scores = np.dot(karar_embeddings, query_embedding)
            if np.max(karar_scores) > 0.7:  # Karar eşleşmesi yeterince yüksekse
                top_karar_indices = np.argsort(karar_scores)[::-1][:top_k]
                return [karar_items[i] for i in top_karar_indices]

        # Karar eşleşmesi yoksa veya yeterince yüksek değilse, tüm öğeleri kontrol et
        embeddings = np.array([m.get("embedding", [0.0] * 768) for m in all_items])
        dot_products = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(dot_products)[::-1][:top_k]

        return [all_items[i] for i in top_indices]

    def update_pinned_context(self, query, response):
        # find article references
        madde_match = re.search(r"(Madde|GEÇİCİ MADDE) ?\d+", query, re.IGNORECASE) or \
                      re.search(r"(Madde|GEÇİCİ MADDE) ?\d+", response, re.IGNORECASE)
        #case references
        case_match = re.search(r"Esas No: ?\d+/\d+", query) or \
                     re.search(r"Esas No: ?\d+/\d+", response)
        karar_match = re.search(r"Karar No: ?\d+/\d+", query) or \
                      re.search(r"Karar No: ?\d+/\d+", response)
        #law references
        law_match = re.search(r"Kanun No: ?\d+", query) or \
                    re.search(r"Kanun No: ?\d+", response)
        #court references
        court_match = re.search(r"(Yargıtay|Danıştay)", query, re.IGNORECASE) or \
                      re.search(r"(Yargıtay|Danıştay)", response, re.IGNORECASE)

        # Pin the most specific context found
        if madde_match:
            self.pinned_context = madde_match.group(0)
        elif case_match:
            self.pinned_context = case_match.group(0)
        elif karar_match:
            self.pinned_context = karar_match.group(0)
        elif law_match:
            self.pinned_context = law_match.group(0)
        elif court_match:
            self.pinned_context = court_match.group(0)
        # else: keep the previous pinned_context

    def summarize_chunk(self, chunk):
        """Bir chunk'ı özetler."""
        history_text = ""
        for turn in chunk:
            if turn["role"] == "user":
                history_text += f"KULLANICI: {turn['content']}\n"
            else:
                history_text += f"BOT: {turn['content']}\n"

        summary_prompt = (
            "Aşağıda bir kullanıcı ile bir hukuk sohbet botu arasındaki konuşma parçası var. "
            "Bu kısmın özetini, önemli hukuki konuları ve tartışılan maddeleri vurgulayarak kısa ve açık şekilde yaz:\n\n"
            f"{history_text}\n\nKISA ÖZET:"
        )

        try:
            response = self.model.generate_content(
                summary_prompt,
                generation_config={
                    "max_output_tokens": 200,
                    "temperature": 0.1
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Özet üretirken hata: {e}")
            return ""

    def add_to_history(self, role, content):
        """Sohbet geçmişine mesaj ekler ve chunk yönetimini yapar."""
        self.history.append({"role": role, "content": content})
        # Chunk dolduysa özetle ve temizle
        if len(self.history) >= self.chunk_size:
            chunk = self.history[:self.chunk_size]
            summary = self.summarize_chunk(chunk)
            self.summaries.append(summary)
            self.history = self.history[self.chunk_size:]

    def build_prompt(self, query, retrieved_items):
        """Chunking sistemini kullanarak prompt oluşturur."""
        # Pinned context
        pinned_prompt = f"ÖNEMLİ KONU: {self.pinned_context}\n" if self.pinned_context else ""

        # Chunk özetleri
        summaries_prompt = ""
        for i, summary in enumerate(self.summaries):
            summaries_prompt += f"SOHBET ÖZETİ {i+1}: {summary}\n"

        # Son chunk (detaylı)
        current_prompt = ""
        for turn in self.history:
            if turn["role"] == "user":
                current_prompt += f"KULLANICI: {turn['content']}\n"
            else:
                current_prompt += f"BOT: {turn['content']}\n"

        # Kaynaklar
        sources_prompt = ""
        if retrieved_items:
            for i, item in enumerate(retrieved_items):
                if "court" in item:
                    sources_prompt += f"{i + 1}. {item['court']} Kararı ({item['tarih']}, Esas No: {item['esas_no']}) - {item['content'][:200]}...\n"
                else:
                    sources_prompt += f"{i + 1}. {self.kanun_data['kanun_adi']} {item['title']} - {item['content'][:200]}...\n"

        # Soru analizi ve özel yönergeler
        question_analysis = self._analyze_question(query)
        
        # Özel durumlar için yönergeler
        special_instructions = ""
        if "garanti" in query.lower():
            special_instructions = """
            ÖNEMLİ GARANTİ BİLGİLERİ:
            1. Garanti belgesi olmasa bile tüketici hakları geçerlidir
            2. Ayıplı mal durumunda garanti şartı aranmaz
            3. Satıcının garanti olmadığı iddiası geçersizdir
            4. Tüketici Kanunu'ndaki seçimlik haklar garanti belgesinden bağımsızdır
            """
        elif "indirimli" in query.lower():
            special_instructions = """
            ÖNEMLİ İNDİRİMLİ ÜRÜN BİLGİLERİ:
            1. İndirimli ürünler de tüketici haklarından yararlanır
            2. İndirim nedeni ayıp değilse değişim/onarım hakkı vardır
            3. Satıcının indirimli ürün gerekçesi geçersizdir
            4. Ayıplı mal durumunda indirimli olması önemli değildir
            """

        return (
            f"{pinned_prompt}"
            f"{summaries_prompt}"
            f"{current_prompt}"
            f"\nSON SORU: '{query}'\n"
            f"{question_analysis}\n"
            f"{special_instructions}\n"
            f"İLGİLİ KAYNAKLAR:\n"
            f"{sources_prompt}"
            "\n"
            "ÖNEMLİ: Aşağıdaki kurallara kesinlikle uy:\n\n"
            "1. Sadece yukarıda verilen kanun maddeleri ve mahkeme kararlarına dayanarak yanıt ver\n"
            "2. Her yanıtın başında hangi kaynakları kullandığını belirt\n"
            "3. Soruya özel ve detaylı yanıt ver, genel bilgileri tekrarlama\n"
            "4. Kesinlikle kaynaklarda olmayan bilgileri ekleme veya varsayımda bulunma\n"
            "5. Kanun maddeleriyle mahkeme kararlarını karşılaştır (eğer varsa)\n"
            "6. Her yanıtın sonunda kullanılan kaynakları tekrar listele\n"
            "7. Yanıtı tamamla, yarım bırakma\n"
            "8. Önceki yanıtları tekrarlama, her soruya özel yanıt ver\n\n"
            "Yanıt formatı:\n"
            "YANIT:\n"
            "[Yanıt metni]\n\n"
            "KAYNAKLAR:\n"
            "[Kullanılan kaynakların listesi]"
        )

    def _analyze_question(self, query):
        """Soruyu analiz eder ve özel yönergeler oluşturur."""
        analysis = "Soru analizi:\n"
        
        # Soru türünü belirle
        if "garanti" in query.lower():
            analysis += "1. Soru türü: Garanti ve yasal haklar\n"
            analysis += "2. Ana konu: Garanti belgesi olmadan tüketici hakları\n"
            analysis += "3. Özel durum: Garanti şartı aranmadan tüketici hakları\n"
        elif "indirimli" in query.lower():
            analysis += "1. Soru türü: İndirimli ürün ve tüketici hakları\n"
            analysis += "2. Ana konu: İndirimli ürünlerde değişim/onarım hakkı\n"
            analysis += "3. Özel durum: İndirimli ürünlerde tüketici hakları\n"
        else:
            analysis += "1. Soru türü: Genel tüketici hakları\n"
            analysis += "2. Ana konu: Ayıplı mal ve tüketici hakları\n"
            analysis += "3. Özel durum: Standart tüketici hakları\n"
        
        return analysis

    def generate_response(self, query):
        """Soruya yanıt üretir ve chunking sistemini kullanır."""
        self.add_to_history("user", query)
        
        # Yeni yanıt üret
        items = self.find_best_passage(query)
        prompt = self.build_prompt(query, retrieved_items=items)
        
        try:
            # İlk yanıt denemesi
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1000,
                    "temperature": 0.7,  # Daha çeşitli yanıtlar için sıcaklığı artırdık
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            response_text = response.text
            
            # Yanıtın tam olup olmadığını kontrol et
            if not self._is_complete_response(response_text):
                # Yanıtı tamamla
                completion_prompt = f"""
                Önceki yanıt: {response_text}
                
                Bu yanıtı tamamla. Yanıt şu bölümleri içermeli:
                1. Ana yanıt (tamamlanmış olmalı)
                2. KAYNAKLAR: bölümü (kullanılan tüm kaynakları listele)
                
                Önemli:
                - Yanıtı yarım bırakma
                - Tüm maddeleri tamamla
                - Kaynakları mutlaka listele
                - Garanti, indirimli ürün gibi özel durumlara değin
                - Önceki yanıtları tekrarlama
                - Her soruya özel yanıt ver
                
                Tamamlanmış yanıt:
                """
                
                completion_response = self.model.generate_content(
                    completion_prompt,
                    generation_config={
                        "max_output_tokens": 1000,
                        "temperature": 0.7
                    }
                )
                response_text = completion_response.text
            
            # Yanıtın hala tam olup olmadığını kontrol et
            if not self._is_complete_response(response_text):
                # Son bir deneme daha yap
                final_prompt = f"""
                Yanıt hala tamamlanmamış. Lütfen şu yanıtı tamamla:
                
                {response_text}
                
                Özellikle şu noktalara dikkat et:
                1. Tüm maddeleri numaralandır ve tamamla
                2. Garanti ve indirimli ürün konularına değin
                3. Kaynakları listele
                4. Yanıtı noktalı virgül veya başka bir işaretle yarım bırakma
                5. Önceki yanıtları tekrarlama
                6. Her soruya özel yanıt ver
                
                Tamamlanmış yanıt:
                """
                
                final_response = self.model.generate_content(
                    final_prompt,
                    generation_config={
                        "max_output_tokens": 1000,
                        "temperature": 0.7
                    }
                )
                response_text = final_response.text
            
        except Exception as e:
            return f"Yanıt üretirken hata: {e}"
        
        self.add_to_history("bot", response_text)
        self.update_pinned_context(query, response_text)
        return response_text

    def _is_complete_response(self, response_text):
        """Yanıtın tam olup olmadığını kontrol eder."""
        # Yanıt boş veya çok kısa ise
        if not response_text or len(response_text.strip()) < 100:
            return False
            
        # Yanıt yarım kalmış mı kontrol et
        if response_text.strip().endswith((",", ";", ":", "-", "*", "1.", "2.", "3.", "4.", "5.")):
            return False
            
        # Kaynaklar bölümü var mı kontrol et
        if "KAYNAKLAR:" not in response_text:
            return False
            
        # Son madde numarasından sonra içerik var mı kontrol et
        last_number = 0
        for line in response_text.split("\n"):
            if line.strip().startswith(str(last_number + 1) + "."):
                last_number += 1
                
        if last_number > 0 and not response_text.split(str(last_number) + ".")[-1].strip():
            return False
            
        return True


def main():
    print("🤖 Kanun ve Karar Chatbot'una hoş geldiniz! 📄")
    print("Çıkmak için 'quit' yazın.\n")

    chatbot = KanunRAGChatbot()

    while True:
        query = input("Soru: ").strip()

        if query.lower() in ['quit', 'exit', 'bye']:
            print("Görüşmek üzere! 👋")
            break

        if not query:
            continue

        print("\n🤖 Düşünüyorum...\n")
        response = chatbot.generate_response(query)
        print("Yanıt:", response, "\n")


if __name__ == "__main__":
    main()