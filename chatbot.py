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
                match = re.search(r"Esas No: (\d+/\d+).*Karar No: (\d+/\d+).*Tarih: (\d+\.\d+\.\d+)", text, re.DOTALL)
                madde_match = re.search(r"6502.*?MADDE (\d+)", text, re.IGNORECASE)

                karar = {
                    "id": f"{'YARGITAY' if 'Yargıtay' in text else 'DANISTAY'}-{match.group(1).replace('/', '-')}" if match else f"KARAR-{len(kararlar)}",
                    "court": "Yargıtay" if "Yargıtay" in text else "Danıştay",
                    "esas_no": match.group(1) if match else "",
                    "karar_no": match.group(2) if match else "",
                    "tarih": match.group(3) if match else "",
                    "ilgili_madde": f"6502-MADDE-{madde_match.group(1)}" if madde_match else "",
                    "content": text[:2000],
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
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"PDF işlenirken hata: {e}")
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

        return (
            f"{pinned_prompt}"
            f"{summaries_prompt}"
            f"{current_prompt}"
            f"\nSON SORU: '{query}'\n"
            f"İLGİLİ KAYNAKLAR:\n"
            f"{sources_prompt}"
            "\n"
            "ÖNEMLİ: Aşağıdaki kurallara kesinlikle uy:\n\n"
            "1. Sadece yukarıda verilen kanun maddeleri ve mahkeme kararlarına dayanarak (öncelikli olarak kanun maddeleri)yanıt ver\n"
            "2. Her yanıtın başında hangi kaynakları kullandığını belirt\n"
            "4. Kesinlikle kaynaklarda olmayan bilgileri ekleme veya varsayımda bulunma\n"
            "5. Kanun maddeleriyle mahkeme kararlarını karşılaştır (eğer varsa)\n"
            "6. Her yanıtın sonunda kullanılan kaynakları tekrar listele (eğer kaynaklar soruyla alakalı değilse listeleme)\n\n"
            "Yanıt formatı:\n"
            "YANIT:\n"
            "[Yanıt metni]\n\n"
            'kaynak verdiğin bilgileri açıkla (eğer kaynaklar soruyla alakalı değilse açıklama)'

        )

    def generate_response(self, query):
        """Soruya yanıt üretir ve chunking sistemini kullanır."""
        self.add_to_history("user", query)
        items = self.find_best_passage(query)
        prompt = self.build_prompt(query, retrieved_items=items)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.1
                }
            )
            response_text = response.text
            self.add_to_history("bot", response_text)
            self.update_pinned_context(query, response_text)
            return response_text
        except Exception as e:
            return f"Yanıt üretirken hata: {e}"


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