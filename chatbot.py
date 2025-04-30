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
        Genel kanunlar ve mahkeme kararları için RAG Chatbot’u başlatır.
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

    def find_best_passage(self, query, top_k=2):
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

    def generate_response(self, query):
        """Soruya yanıt üretir."""
        items = self.find_best_passage(query)

        prompt = textwrap.dedent(f"""You are a helpful legal bot that answers questions based on Turkish laws and court decisions.
        Respond in a complete, conversational sentence in Turkish. Explain complicated concepts simply and compare with relevant court decisions if applicable.

        SORU: '{query}'
        İLGİLİ KAYNAKLAR:
        """)
        for i, item in enumerate(items):
            if "court" in item:
                prompt += f"{i + 1}. {item['court']} Kararı ({item['tarih']}, Esas No: {item['esas_no']}) - {item['content'][:200]}...\n"
            else:
                prompt += f"{i + 1}. {self.kanun_data['kanun_adi']} {item['title']} - {item['content'][:200]}...\n"

        prompt += f"""
        Bu kaynaklara dayanarak, soruya kısa, anlaşılır ve doğru bir yanıt ver. Yanıtın sonunda ilgili maddeleri ve varsa mahkeme kararlarını kaynak olarak belirt.
        Kanun maddeleriyle mahkeme kararlarını karşılaştır ve kararların maddeyi nasıl yorumladığını kısa ve anlaşılır bir şekilde açıkla.
        Eğer ilgili bir karar yoksa, sadece kanun maddelerine dayanarak yanıt ver.
        
        """
        #Kanun: {self.kanun_data['kanun_adi']} ({self.kanun_data['kanun_numarasi']})

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 500,
                    "temperature": 0.1
                }
            )
            print(f"""Kanun: {self.kanun_data['kanun_adi']} ({self.kanun_data['kanun_numarasi']})""")
            return response.text
        except Exception as e:
            return f"Yanıt üretirken hata: {e}"


def main():
    print("🤖 Kanun ve Karar Chatbot’una hoş geldiniz! 📄")
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