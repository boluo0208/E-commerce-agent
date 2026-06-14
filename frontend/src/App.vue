<template>
  <main class="page">
    <section class="workspace">
      <div class="header">
        <div>
          <p class="eyebrow">Product Content Agent</p>
          <h1>跨境电商商品内容生成</h1>
        </div>
        <el-tag :type="backendReady ? 'success' : 'warning'" effect="light">
          {{ backendReady ? '后端已连接' : '检查后端中' }}
        </el-tag>
      </div>

      <div class="content-grid">
        <section class="form-panel">
          <el-form label-position="top">
            <el-form-item label="中文商品标题">
              <el-input
                v-model="chineseTitle"
                size="large"
                maxlength="120"
                show-word-limit
                placeholder="例如：女士宽松纯色长袖衬衫"
              />
            </el-form-item>

            <el-form-item label="商品图片">
              <el-upload
                class="upload"
                drag
                multiple
                action="#"
                :auto-upload="false"
                :show-file-list="false"
                accept="image/png,image/jpeg,image/webp"
                :on-change="handleFileChange"
              >
                <div class="upload-empty">
                  <span class="upload-icon">+</span>
                  <span>点击或拖入多张商品图片</span>
                </div>
              </el-upload>

              <div v-if="imageFiles.length" class="preview-grid">
                <div v-for="item in imageFiles" :key="item.uid" class="preview-card">
                  <img :src="item.url" :alt="item.name" class="preview-image" />
                  <div class="preview-meta">
                    <span :title="item.name">{{ item.name }}</span>
                    <el-button text type="danger" @click="removeImage(item.uid)">删除</el-button>
                  </div>
                </div>
              </div>
            </el-form-item>

            <el-button
              type="primary"
              size="large"
              class="submit"
              :loading="submitting"
              :disabled="!canSubmit"
              @click="generateExport"
            >
              生成并下载 ZIP
            </el-button>
          </el-form>
        </section>

        <section class="status-panel">
          <h2>处理流程</h2>
          <el-steps direction="vertical" :active="activeStep" finish-status="success">
            <el-step title="上传中文标题和多张图片" />
            <el-step title="Mimo 逐张识别商品图片" />
            <el-step title="DeepSeek 逐条生成多语言文案" />
            <el-step title="Pillow 批量处理 660×900 图片" />
            <el-step title="导出 Excel 并打包 ZIP" />
          </el-steps>

          <div class="summary">
            <span>已选择图片</span>
            <strong>{{ imageFiles.length }}</strong>
          </div>

          <el-alert
            v-if="message"
            class="message"
            :title="message"
            :type="messageType"
            show-icon
            :closable="false"
          />
        </section>
      </div>
    </section>
  </main>
</template>

<script setup>
import axios from 'axios'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

const API_BASE_URL = 'http://127.0.0.1:8000'

const chineseTitle = ref('')
const imageFiles = ref([])
const submitting = ref(false)
const backendReady = ref(false)
const activeStep = ref(0)
const message = ref('')
const messageType = ref('info')

const canSubmit = computed(() => chineseTitle.value.trim() && imageFiles.value.length > 0 && !submitting.value)

onMounted(async () => {
  try {
    await axios.get(`${API_BASE_URL}/health`, { timeout: 3000 })
    backendReady.value = true
  } catch {
    backendReady.value = false
    message.value = '后端暂时不可用，请确认 FastAPI 已运行在 8000 端口。'
    messageType.value = 'warning'
  }
})

onBeforeUnmount(() => {
  imageFiles.value.forEach((item) => URL.revokeObjectURL(item.url))
})

function handleFileChange(uploadFile) {
  const file = uploadFile.raw
  if (!file) return

  const exists = imageFiles.value.some(
    (item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified,
  )
  if (exists) return

  imageFiles.value.push({
    uid: uploadFile.uid,
    name: file.name,
    size: file.size,
    lastModified: file.lastModified,
    file,
    url: URL.createObjectURL(file),
  })
  activeStep.value = 1
  message.value = ''
}

function removeImage(uid) {
  const item = imageFiles.value.find((image) => image.uid === uid)
  if (item) URL.revokeObjectURL(item.url)
  imageFiles.value = imageFiles.value.filter((image) => image.uid !== uid)
}

async function generateExport() {
  if (!canSubmit.value) return

  submitting.value = true
  activeStep.value = 2
  message.value = `正在生成 ${imageFiles.value.length} 张图片的内容，请稍等...`
  messageType.value = 'info'

  const formData = new FormData()
  formData.append('chinese_title', chineseTitle.value.trim())
  imageFiles.value.forEach((item) => {
    formData.append('images', item.file)
  })

  try {
    const response = await axios.post(`${API_BASE_URL}/api/products/generate`, formData, {
      responseType: 'blob',
      timeout: 300000,
    })

    activeStep.value = 5
    downloadBlob(response.data, 'product_export.zip')
    message.value = '生成完成，ZIP 已开始下载。'
    messageType.value = 'success'
    ElMessage.success('生成完成')
  } catch (error) {
    activeStep.value = 1
    const detail = await readErrorDetail(error)
    message.value = detail || '生成失败，请查看后端日志。'
    messageType.value = 'error'
    ElMessage.error(message.value)
  } finally {
    submitting.value = false
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

async function readErrorDetail(error) {
  const data = error?.response?.data
  if (data instanceof Blob) {
    try {
      const text = await data.text()
      const parsed = JSON.parse(text)
      return parsed.detail || text
    } catch {
      return ''
    }
  }
  return error?.message || ''
}
</script>
